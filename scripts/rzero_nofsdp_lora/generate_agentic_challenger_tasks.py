# -*- coding: utf-8 -*-
"""
Agentic Challenger: generate NEW machine-checkable tasks.

This is the agentic analogue of the challenger. It does NOT resample fixed
problems: it calls ``env.sample_new_task(rng, difficulty)`` to GENERATE genuinely new
tasks (the round-4 fix), then GATES each task on two verifier-grounded criteria:

  1. Well-posedness:  ``env.is_well_posed(task)`` -- the task is solvable AND verifiable.
  2. Learnability band: run the solver policy ``probe_k`` times via the SAME multi-turn
     loop the rollout builder uses (rollout_agentic.run_episode), measure the solve-rate
     (fraction of probe episodes whose terminal verifier reward is a full success), and
     keep ONLY tasks whose solve-rate falls inside ``[score_min, score_max]`` -- the
     edge-of-ability "learnable" band. Too easy (>score_max) or too hard (<score_min)
     tasks are rejected.

Difficulty-suppression: each accepted task costs at most ``probe_k`` solver episodes;
the search caps the number of generation attempts (``--max_attempts`` / a multiple of
``num_tasks``) and rejects DEGENERATE tasks (not well-posed, empty predicate library,
or no items / no constraints) so the loop terminates even when the band is hard to hit.

Output: ``challenger_tasks.jsonl`` -- one serialized Task (Task.to_dict()) per line,
each augmented with a ``learnability`` block in ``info`` (solve_rate, probe_k, rewards).
The tasks are then SPLIT by difficulty into D1 (train) and D2 (held-out eval), written
as ``challenger_tasks.D1.jsonl`` / ``challenger_tasks.D2.jsonl`` next to the main file,
plus a ``summary.json``.

DRY-RUN constraint: with ``--dry_run`` (or ``--policy random``) the whole
path is PURE PYTHON -- env=mock + RandomPolicy stand-in solver -- and runs on a CPU box
with NO torch and NO vllm installed. Therefore this module imports NEITHER torch NOR vllm
at module top; the vLLM solver policy is built lazily ONLY when ``--policy vllm`` is used
(rollout_agentic.VLLMPolicy lazy-imports vllm inside its own __init__).

Robust import:
    import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from verifiers.base import ...; from envs.base import ...; from roles import ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Band metric for the challenger: "solve_rate" (binary full-success, R-Zero default)
# or "mean" (continuous partial credit). On HARD benchmarks (DeepPlanning: frontier LLMs <20% full
# success) the binary band finds NO in-band tasks; "mean" tracks the learnable frontier in partial-
# reward space so self-play can proceed.
_BAND_METRIC = os.environ.get("RZERO_BAND_METRIC", "solve_rate").lower()

# --- Robust import: make sibling modules importable
# regardless of the caller's cwd, BEFORE importing them. ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# REUSED UNCHANGED (frozen credit machinery). Imported here as the contract anchor;
# the challenger itself only produces Tasks, but importing fails loud if the reused
# package is missing. Pure-python, no torch.
from verifiers.base import StageRecord  # noqa: F401  (contract anchor)

# Frozen agentic env contracts.
from envs.base import Task, AgenticEnv  # noqa: E402

# Roles + multi-turn loop + dry-run policy. rollout_agentic does NOT import torch/vllm
# at module top (VLLMPolicy lazy-imports vllm inside its own __init__), so importing
# these here is safe on a CPU box with neither installed.
from roles import Planner, Executor  # noqa: E402
from rollout_agentic import RandomPolicy, run_episode, Trajectory  # noqa: E402


# A full success in [0,1] terminal verifier reward. The mock env returns exactly 1.0
# when ALL hard constraints + the target predicate are satisfied at finalize; partial
# credit is < 1.0. We treat "solved" as terminal_reward >= SOLVE_THRESHOLD.
SOLVE_THRESHOLD = 1.0 - 1e-9


# --------------------------------------------------------------------------- #
# Env construction                                                              #
# --------------------------------------------------------------------------- #
def build_env(args: argparse.Namespace) -> AgenticEnv:
    """Construct the agentic environment for the requested ``--env``.

    One of the wired real benchmarks: textcraft / scienceworld / alfworld / herobench / longreason.
    """
    if args.env == "textcraft":
        # CRAFTING benchmark: prerequisite recipe DAG -> dependency-rich long horizon; the
        # substrate for PATSP (ordering/premature/missing failures + recovery). See textcraft_env.py.
        from envs.textcraft_env import TextCraftEnv
        return TextCraftEnv(seed=args.seed)
    if args.env == "scienceworld":
        from envs.scienceworld_env import ScienceWorldEnv
        return ScienceWorldEnv(seed=args.seed)
    if args.env == "alfworld":
        from envs.alfworld_env import ALFWorldEnv
        return ALFWorldEnv(seed=args.seed)
    if args.env == "herobench":
        from envs.herobench_env import HeroBenchEnv
        return HeroBenchEnv(seed=args.seed)
    if args.env == "longreason":
        from envs.longreason_env import LongReasonEnv
        return LongReasonEnv(seed=args.seed)

    raise ValueError(f"unknown --env {args.env!r}")


# --------------------------------------------------------------------------- #
# Solver policy factory (stand-in solver used to measure learnability)          #
# --------------------------------------------------------------------------- #
def make_solver_policies(
    args: argparse.Namespace,
    env: AgenticEnv,
    sub_seed: int,
) -> Tuple[Any, Any]:
    """Return (planner_policy, executor_policy) used as the LEARNABILITY PROBE solver.

    --policy random (or --dry_run): RandomPolicy stand-in solver -- env-aware, pure python,
        no torch/vllm. This is the spec's dry-run stand-in for "the current Executor".
    --policy vllm: build VLLMPolicy for each role. vLLM is lazy-imported INSIDE VLLMPolicy
        (never at module top), so this branch is only exercised when a model is requested.

    Distinct sub-seeds per probe episode make the probe runs differ (the solve-rate is a
    Monte-Carlo estimate of task learnability under the current solver).
    """
    if args.policy == "random" or args.dry_run:
        planner_policy = RandomPolicy(env, task=None, seed=1000 + sub_seed)
        executor_policy = RandomPolicy(env, task=None, seed=2000 + sub_seed)
        return planner_policy, executor_policy

    if args.policy == "vllm":
        # Lazy: importing VLLMPolicy is torch/vllm-free; constructing it lazy-imports vllm.
        from rollout_agentic import VLLMPolicy

        # FULL-PARAM mode: two independent full checkpoints -> two SEPARATE engines (no LoRA).
        # Cached by (planner_model, executor_model) so probe episodes reuse them (same OOM-churn
        # reason as the LoRA path below). Both engines share the one visible card -> halved budget.
        if args.planner_model and args.executor_model:
            fcache = make_solver_policies.__dict__.setdefault("_full_engine_cache", {})
            key = (args.planner_model, args.executor_model)
            fshared = fcache.get(key)
            if fshared is None:
                planner_policy = VLLMPolicy(
                    role="planner", lora=None, model=args.planner_model,
                    max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p,
                    gpu_memory_utilization=args.gpu_memory_utilization, enable_lora=False,
                )
                executor_policy = VLLMPolicy(
                    role="executor", lora=None, model=args.executor_model,
                    max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p,
                    gpu_memory_utilization=args.gpu_memory_utilization, enable_lora=False,
                )
                fcache[key] = (planner_policy.llm, getattr(planner_policy, "tokenizer", None),
                               executor_policy.llm, getattr(executor_policy, "tokenizer", None))
            else:
                p_llm, p_tok, e_llm, e_tok = fshared
                planner_policy = VLLMPolicy(role="planner", lora=None, llm=p_llm, tokenizer=p_tok,
                                            max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)
                executor_policy = VLLMPolicy(role="executor", lora=None, llm=e_llm, tokenizer=e_tok,
                                             max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)
            return planner_policy, executor_policy

        if not args.base_model:
            raise ValueError("--policy vllm requires --base_model <path/hf-id>.")
        # Two role adapters on the shared base (planner int_id=41, executor int_id=42).
        # The planner adapter is optional for the probe; reuse executor adapter for both
        # roles if only one is given (the probe just needs a solver, not the final policy).
        planner_lora = args.planner_lora or args.executor_lora
        executor_lora = args.executor_lora or args.planner_lora
        # ENGINE REUSE (critical): make_solver_policies is called ONCE PER PROBE EPISODE.
        # Without a cache, every call spun up a brand-new vLLM engine for the base model
        # while the PREVIOUS episode's engine still held its gpu_memory_utilization share
        # -> the 2nd episode OOMs ("Free memory < desired utilization") and the probe
        # never advances (plus catastrophic reload churn). Load the base engine ONCE and
        # reuse it across all episodes/tasks; per-role LoRA is served from the one engine
        # via distinct LoRARequests (planner int_id=41, executor int_id=42).
        cache = make_solver_policies.__dict__.setdefault("_engine_cache", {})
        shared = cache.get(args.base_model)
        if shared is None:
            planner_policy = VLLMPolicy(
                role="planner",
                lora=planner_lora,
                model=args.base_model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            shared_llm = getattr(planner_policy, "llm", None)
            shared_tok = getattr(planner_policy, "tokenizer", None)
            cache[args.base_model] = (shared_llm, shared_tok)
        else:
            shared_llm, shared_tok = shared
            planner_policy = VLLMPolicy(
                role="planner",
                lora=planner_lora,
                llm=shared_llm,
                tokenizer=shared_tok,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        # Executor shares the SAME engine (distinct role/LoRA, one base model on GPU).
        executor_policy = VLLMPolicy(
            role="executor",
            lora=executor_lora,
            llm=shared_llm,
            tokenizer=shared_tok,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        return planner_policy, executor_policy

    raise ValueError(f"unknown --policy {args.policy!r}; expected 'random' or 'vllm'.")


# --------------------------------------------------------------------------- #
# Learnability probe                                                            #
# --------------------------------------------------------------------------- #
def probe_learnability(
    env: AgenticEnv,
    task: Task,
    args: argparse.Namespace,
    attempt_seed: int,
) -> Dict[str, Any]:
    """Run the solver policy ``probe_k`` times on ``task`` and measure the solve-rate.

    Uses rollout_agentic.run_episode -- the SAME multi-turn loop the rollout builder uses
    -- so the learnability estimate matches the regime the task will actually be trained in.
    A probe episode "solves" the task iff its terminal verifier reward >= SOLVE_THRESHOLD
    (full programmatic success). Returns a dict with solve_rate + per-episode rewards.
    """
    rewards: List[float] = []
    solves = 0
    for k in range(args.probe_k):
        planner_policy, executor_policy = make_solver_policies(
            args, env, sub_seed=attempt_seed * 1000 + k
        )
        traj: Trajectory = run_episode(
            env=env,
            task=task,
            planner=Planner,
            executor=Executor,
            planner_policy=planner_policy,
            executor_policy=executor_policy,
            max_steps=args.max_steps,
            max_replans=args.max_replans,
            sample_index=k,
        )
        r = float(traj.terminal_reward)
        rewards.append(round(r, 6))
        if r >= SOLVE_THRESHOLD:
            solves += 1

    solve_rate = (solves / float(args.probe_k)) if args.probe_k > 0 else 0.0
    return {
        "solve_rate": solve_rate,
        "n_solved": solves,
        "probe_k": int(args.probe_k),
        "rewards": rewards,
        "mean_reward": (sum(rewards) / len(rewards)) if rewards else 0.0,
        "solve_threshold": SOLVE_THRESHOLD,
    }


# --------------------------------------------------------------------------- #
# Degeneracy / well-posedness gates                                             #
# --------------------------------------------------------------------------- #
def is_degenerate(env: AgenticEnv, task: Task) -> Tuple[bool, str]:
    """Reject DEGENERATE tasks (difficulty-suppression). Returns (degenerate, reason).

    A task is degenerate if it has no predicate library (nothing to verify), no
    actionable content, or its predicate library is empty -- such tasks cannot yield a
    meaningful learnability signal. This is intentionally conservative and env-agnostic.
    """
    try:
        pred_lib = list(env.predicate_library(task))
    except Exception as exc:
        return True, f"predicate_library raised: {exc!r}"
    if not pred_lib:
        return True, "empty predicate_library (nothing verifiable)"
    # Mock-env-style content check (skipped gracefully if the field is absent for other envs).
    spec = task.spec or {}
    if isinstance(spec, dict) and "items" in spec and not spec.get("items"):
        return True, "no items in spec (nothing to select)"
    if not (task.constraints or {}):
        return True, "no constraints (nothing to satisfy)"
    return False, ""


# --------------------------------------------------------------------------- #
# Main generation loop                                                          #
# --------------------------------------------------------------------------- #
def generate_tasks(env: AgenticEnv, args: argparse.Namespace) -> Tuple[List[Task], Dict[str, Any]]:
    """Generate, gate, and collect up to ``num_tasks`` learnable, well-posed tasks.

    Returns (accepted_tasks, stats). Each accepted Task carries a ``learnability`` block
    plus generation provenance in task.info.
    """
    rng = random.Random(args.seed)
    accepted: List[Task] = []

    # Difficulty-suppression: cap the number of generation attempts so the search always
    # terminates even when the learnability band is hard to hit.
    max_attempts = args.max_attempts
    if max_attempts is None or max_attempts <= 0:
        max_attempts = max(args.num_tasks * args.attempt_multiplier, args.num_tasks + 8)

    stats = {
        "attempts": 0,
        "accepted": 0,
        "rejected_not_well_posed": 0,
        "rejected_degenerate": 0,
        "rejected_too_easy": 0,   # solve_rate > score_max
        "rejected_too_hard": 0,   # solve_rate < score_min
        "max_attempts": int(max_attempts),
    }

    seen_task_ids = set()
    attempt = 0
    while len(accepted) < args.num_tasks and attempt < max_attempts:
        attempt += 1
        stats["attempts"] = attempt

        # GENERATE a new task (not a resample). difficulty in [0,1].
        task = env.sample_new_task(rng, args.difficulty)

        # Guard against an env that returns duplicate task_ids.
        if task.task_id in seen_task_ids:
            task.task_id = f"{task.task_id}-a{attempt:05d}"
        seen_task_ids.add(task.task_id)

        # --- gate 1a: degeneracy (difficulty-suppression) ---
        degen, reason = is_degenerate(env, task)
        if degen:
            stats["rejected_degenerate"] += 1
            if args.verbose:
                print(f"[reject:degenerate] {task.task_id}: {reason}")
            continue

        # --- gate 1b: well-posedness (solvable AND verifiable) ---
        try:
            well = bool(env.is_well_posed(task))
        except Exception as exc:
            # An env-side failure to evaluate well-posedness is treated as not-well-posed
            # (and logged); we do not crash the whole generation run on one bad task.
            well = False
            if args.verbose:
                print(f"[reject:well_posed_error] {task.task_id}: {exc!r}")
        if not well:
            stats["rejected_not_well_posed"] += 1
            if args.verbose:
                print(f"[reject:not_well_posed] {task.task_id}")
            continue

        # --- gate 2: learnability band ---
        learn = probe_learnability(env, task, args, attempt_seed=attempt)
        sr = learn["solve_rate"]

        # mean_reward = continuous partial credit (fraction of predicates sat), distinct
        # from solve_rate (binary full-success rate). Logging it tells us whether a
        # too_hard task is "model acts coherently but misses full success" (mean>0) vs
        # "model can't act at all / output format broken" (mean~=0) -- the key diagnosis.
        mr = learn.get("mean_reward", 0.0)
        # learnability band: accept tasks whose solver score is inside [score_min, score_max]
        # (RZERO_BAND_METRIC: "mean" = continuous partial credit, else binary full-success rate).
        metric = mr if _BAND_METRIC == "mean" else sr
        if metric > args.score_max:
            stats["rejected_too_easy"] += 1
            if args.verbose:
                print(f"[reject:too_easy] {task.task_id}: {_BAND_METRIC}={metric:.3f} > {args.score_max} "
                      f"(solve_rate={sr:.3f} mean_reward={mr:.3f})")
            continue
        if metric < args.score_min:
            stats["rejected_too_hard"] += 1
            if args.verbose:
                print(f"[reject:too_hard] {task.task_id}: {_BAND_METRIC}={metric:.3f} < {args.score_min} "
                      f"(solve_rate={sr:.3f} mean_reward={mr:.3f})")
            continue

        # --- accept ---
        task.info = dict(task.info)
        task.info["learnability"] = learn
        task.info["challenger"] = {
            "env": args.env,
            "difficulty": float(args.difficulty),
            "score_min": float(args.score_min),
            "score_max": float(args.score_max),
            "policy": ("random" if (args.dry_run or args.policy == "random") else args.policy),
            "attempt": attempt,
            "seed": int(args.seed),
        }
        accepted.append(task)
        stats["accepted"] = len(accepted)
        if args.verbose:
            print(f"[accept] {task.task_id}: solve_rate={sr:.3f} "
                  f"(band [{args.score_min},{args.score_max}]) "
                  f"difficulty={task.info.get('difficulty', args.difficulty)}")

    return accepted, stats


# --------------------------------------------------------------------------- #
# D1 / D2 difficulty split                                                      #
# --------------------------------------------------------------------------- #
def split_by_difficulty(tasks: List[Task], threshold: float) -> Tuple[List[Task], List[Task]]:
    """Split tasks into D1 (train) and D2 (held-out eval) by difficulty.

    Tasks with difficulty < threshold -> D1 (train); difficulty >= threshold -> D2 (eval).
    Difficulty is read from task.info["difficulty"] (set by the env's sampler), falling
    back to the requested --difficulty. With a single uniform --difficulty (the common
    case), the split is interleaved by index so BOTH files are non-empty and the held-out
    set is a genuine, disjoint sample of the SAME difficulty (still a valid train/eval
    split for the dry-run plumbing).
    """
    def diff_of(t: Task) -> float:
        try:
            return float((t.info or {}).get("difficulty"))
        except (TypeError, ValueError):
            return float("nan")

    diffs = [diff_of(t) for t in tasks]
    # If difficulties vary, split by the threshold; else interleave by index.
    finite = [d for d in diffs if d == d]  # drop NaN
    varies = len(set(round(d, 6) for d in finite)) > 1 if finite else False

    d1: List[Task] = []
    d2: List[Task] = []
    if varies:
        for t, d in zip(tasks, diffs):
            if d == d and d >= threshold:
                d2.append(t)
            else:
                d1.append(t)
    else:
        # Uniform difficulty: interleave (even index -> D1, odd -> D2) so both are non-empty.
        for i, t in enumerate(tasks):
            (d1 if i % 2 == 0 else d2).append(t)

    return d1, d2


# --------------------------------------------------------------------------- #
# IO                                                                            #
# --------------------------------------------------------------------------- #
def write_jsonl(path: Path, tasks: List[Task]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Agentic Challenger: generate NEW machine-checkable tasks gated by "
                    "well-posedness + learnability band."
    )
    # Environment.
    p.add_argument("--env", choices=["textcraft", "scienceworld", "alfworld", "herobench", "longreason"], default="textcraft",
                   help="Which agentic environment to sample tasks from.")
    p.add_argument("--dp_split", default="train",
                   help="DeepPlanning dataset split selector (ADJUST-ON-SERVER).")

    # Generation controls.
    p.add_argument("--num_tasks", type=int, default=8,
                   help="Number of accepted (well-posed + learnable) tasks to produce.")
    p.add_argument("--difficulty", type=float, default=0.5,
                   help="Difficulty in [0,1] passed to env.sample_new_task.")
    p.add_argument("--max_attempts", type=int, default=0,
                   help="Cap on generation attempts (difficulty-suppression). "
                        "0 = auto (num_tasks * attempt_multiplier).")
    p.add_argument("--attempt_multiplier", type=int, default=12,
                   help="auto max_attempts = num_tasks * this (when --max_attempts=0).")

    # Learnability band.
    p.add_argument("--score_min", type=float, default=0.3,
                   help="Keep tasks with solve-rate >= this (edge-of-ability lower bound).")
    p.add_argument("--score_max", type=float, default=0.8,
                   help="Keep tasks with solve-rate <= this (edge-of-ability upper bound).")
    p.add_argument("--probe_k", type=int, default=6,
                   help="Solver probe episodes per candidate task (learnability MC estimate).")

    # Solver policy (the probe's stand-in solver).
    p.add_argument("--policy", choices=["vllm", "random"], default="random",
                   help="Probe solver policy. 'random' (RandomPolicy) is the dry-run stand-in.")
    p.add_argument("--dry_run", action="store_true",
                   help="Force the pure-python path: env=mock + RandomPolicy, no torch/vllm.")

    # vLLM solver knobs (only used with --policy vllm).
    p.add_argument("--base_model", default=None, help="Base model path/hf-id (for --policy vllm).")
    p.add_argument("--planner_lora", default=None, help="Planner LoRA adapter path (--policy vllm).")
    p.add_argument("--executor_lora", default=None, help="Executor LoRA adapter path (--policy vllm).")
    p.add_argument("--planner_model", default=None, help="FULL-PARAM: full Planner checkpoint (own engine, no LoRA).")
    p.add_argument("--executor_model", default=None, help="FULL-PARAM: full Executor checkpoint (own engine, no LoRA).")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)

    # Episode loop knobs (tunables).
    p.add_argument("--max_steps", type=int, default=16, help="Max executor steps per probe episode.")
    p.add_argument("--max_replans", type=int, default=3, help="Max verifier-event replans per episode.")

    # Split + output.
    p.add_argument("--split_threshold", type=float, default=0.5,
                   help="difficulty >= this -> D2 (held-out eval); else D1 (train). "
                        "Used only when difficulties vary; otherwise an index interleave.")
    p.add_argument("--out_jsonl", required=True,
                   help="Output path for challenger_tasks.jsonl (D1/D2 written alongside).")
    p.add_argument("--seed", type=int, default=202, help="RNG seed (task sampling + probe).")
    p.add_argument("--verbose", action="store_true", help="Log per-candidate accept/reject decisions.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # --dry_run forces the RandomPolicy stand-in solver (no torch/vllm).
    if args.dry_run:
        args.policy = "random"

    if args.score_min > args.score_max:
        raise ValueError(f"--score_min ({args.score_min}) must be <= --score_max ({args.score_max}).")

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[challenger] env={args.env} policy={'random' if args.dry_run else args.policy} "
          f"num_tasks={args.num_tasks} difficulty={args.difficulty} "
          f"band=[{args.score_min},{args.score_max}] probe_k={args.probe_k} seed={args.seed}")

    env = build_env(args)

    accepted, stats = generate_tasks(env, args)

    # D1 (train) / D2 (held-out eval) split by difficulty.
    d1, d2 = split_by_difficulty(accepted, args.split_threshold)

    # Tag the split into each task's info (so downstream stages can read it).
    for t in d1:
        t.info = dict(t.info); t.info["split"] = "D1"
    for t in d2:
        t.info = dict(t.info); t.info["split"] = "D2"

    # Write outputs.
    write_jsonl(out_path, accepted)
    d1_path = out_path.with_name(out_path.stem + ".D1" + out_path.suffix)
    d2_path = out_path.with_name(out_path.stem + ".D2" + out_path.suffix)
    write_jsonl(d1_path, d1)
    write_jsonl(d2_path, d2)

    summary = {
        "env": args.env,
        "policy": ("random" if args.dry_run else args.policy),
        "num_tasks_requested": int(args.num_tasks),
        "num_tasks_accepted": len(accepted),
        "difficulty": float(args.difficulty),
        "score_min": float(args.score_min),
        "score_max": float(args.score_max),
        "probe_k": int(args.probe_k),
        "seed": int(args.seed),
        "stats": stats,
        "split": {"D1": len(d1), "D2": len(d2), "split_threshold": float(args.split_threshold)},
        "out_jsonl": str(out_path),
        "d1_jsonl": str(d1_path),
        "d2_jsonl": str(d2_path),
        "solve_rates": [t.info.get("learnability", {}).get("solve_rate") for t in accepted],
    }
    summary_path = out_path.with_name(out_path.stem + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[challenger] accepted {len(accepted)}/{args.num_tasks} tasks "
          f"in {stats['attempts']} attempts "
          f"(too_easy={stats['rejected_too_easy']} too_hard={stats['rejected_too_hard']} "
          f"not_well_posed={stats['rejected_not_well_posed']} degenerate={stats['rejected_degenerate']})")
    print(f"[challenger] D1(train)={len(d1)} D2(eval)={len(d2)}")
    print(f"[save] tasks   = {out_path}")
    print(f"[save] D1      = {d1_path}")
    print(f"[save] D2      = {d2_path}")
    print(f"[save] summary = {summary_path}")

    if not accepted:
        print("[challenger][WARN] no tasks accepted. Loosen the band (--score_min/--score_max), "
              "lower --difficulty, raise --max_attempts, or increase --probe_k.")
        # Non-zero so an orchestrator can detect an empty challenger output.
        return 2
    return 0


if __name__ == "__main__":
    rc = main()
    # ScienceWorld (py4j JVM) / ALFWorld (textworld) leave non-daemon threads that hang interpreter
    # shutdown AFTER all outputs are written -> the self-play driver waits on this subprocess forever
    # (observed: gen finished+saved at 16:11, process idled at 0.2% CPU for 90 min). Force-exit once
    # main() has returned: outputs are already flushed to disk; preserve main()'s exit code (2 = empty
    # challenger output, which the orchestrator checks).
    # CRITICAL companion: _exit alone ORPHANS our vLLM EngineCore / java children, which keep their
    # ~40GB GPU allocations for many minutes -> the NEXT stage's engine init OOMs ("Free memory on
    # device 12/95 GiB", killed the lr2e6 PBS run at r2). Kill direct children FIRST, then _exit.
    import os as _os, sys as _sys, glob as _glob, signal as _signal
    _me = _os.getpid()
    for _stat in _glob.glob("/proc/[0-9]*/stat"):
        try:
            _data = open(_stat).read()
            _after = _data[_data.rindex(")") + 2:].split()   # robust to spaces in comm
            if int(_after[1]) == _me:                        # ppid == us -> our direct child
                _os.kill(int(_stat.split("/")[2]), _signal.SIGKILL)
        except Exception:
            pass
    _sys.stdout.flush(); _sys.stderr.flush()
    _os._exit(int(rc) if isinstance(rc, int) else 0)
