# -*- coding: utf-8 -*-
"""
Agentic rollout driver -- emits agentic_trajectories.jsonl.

This is the agentic analogue of build_solver_grpo_rollouts.py: a DIFFERENT rollout
generator that drives the multi-turn Planner/Executor loop (rollout_agentic.run_episode)
over an AgenticEnv and writes one JSONL row per (task_id, rollout sample). The row schema
is FROZEN:

    {task_id, sample_index, terminal_reward, turns:[Turn...], predicate_library}

(We also pass through `stage_records_global` and `info` from Trajectory.to_dict() --
extra keys that split_agentic_rollouts_by_role.py simply does not read; the five
required keys are always present.)

Group structure: for each task we run `group_size` episodes that share the
same uid (=task_id) so the downstream role-split can compute group-relative advantage
over same-task rollouts of terminal_reward.

POLICIES:
  - --dry_run  OR  --policy random : RandomPolicy (env-aware, CPU, NO model). This path
    imports NO torch and NO vllm. It is the runnable reference for the whole stack.
  - --policy vllm                  : VLLMPolicy per role (planner/executor) with optional
    per-role LoRA adapters. vLLM is LAZY-imported inside VLLMPolicy ONLY (never at module
    top here), so this file imports fine on a CPU box without vllm; only --policy vllm
    actually touches it.

DeepPlanning (--env deepplanning) is the documented real-benchmark adapter; its dataset /
verifier bindings are marked ADJUST-ON-SERVER inside envs/deepplanning.py and FAIL LOUD if
absent. The mock env (--env mock) is the fully-runnable CPU reference.

Robust import: sys.path is fixed so `from verifiers.base import ...`,
`from envs ... import ...`, `from roles import ...` and `from rollout_agentic import ...`
resolve regardless of cwd. torch / vllm are NEVER imported at module top.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Robust import: make sibling modules importable
# regardless of the caller's cwd. NO torch / NO vllm imported at module top so the
# mock-env + RandomPolicy dry-run path runs on a bare CPU box.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.base import Task  # noqa: E402  (FROZEN dataclass)
from roles import Planner, Executor  # noqa: E402  (role prompt builders + parsers)
from rollout_agentic import (  # noqa: E402
    RandomPolicy,
    VLLMPolicy,  # lazy-imports vllm only inside its own __init__/act
    run_episode,
)


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run agentic Planner/Executor rollouts -> agentic_trajectories.jsonl.",
    )
    p.add_argument(
        "--env",
        choices=["textcraft", "scienceworld", "alfworld", "herobench", "longreason"],
        default="textcraft",
        help="Environment: one of the four wired real benchmarks "
        "(textcraft / scienceworld / alfworld / herobench / longreason).",
    )
    p.add_argument(
        "--group_size",
        type=int,
        default=4,
        help="Episodes (rollout samples) per task; shared uid=task_id for group-relative "
        "advantage.",
    )
    p.add_argument(
        "--num_tasks",
        type=int,
        default=4,
        help="Number of tasks to roll out (when --tasks_jsonl is not given). With "
        "--tasks_jsonl, caps how many of the provided tasks are used (<=0 means all).",
    )
    p.add_argument(
        "--tasks_jsonl",
        default=None,
        help="Optional path to challenger_tasks.jsonl (Task serialized, one per line). "
        "When given, tasks come from this file instead of env.list_tasks/sample_new_task.",
    )
    p.add_argument(
        "--max_steps",
        type=int,
        default=16,
        help="Max Executor actions per episode (multi-turn loop horizon).",
    )
    p.add_argument(
        "--max_replans",
        type=int,
        default=3,
        help="Max verifier-EVENT replans (boundary 0..max_replans) per episode.",
    )
    p.add_argument(
        "--out_jsonl",
        required=True,
        help="Output path for agentic_trajectories.jsonl (one row per (task, sample)).",
    )
    p.add_argument(
        "--policy",
        choices=["vllm", "random"],
        default="random",
        help="'random' = RandomPolicy (CPU, no model); 'vllm' = VLLMPolicy per role "
        "(lazy vllm).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Force the RandomPolicy + (default) mock env path: NO torch / NO vllm. "
        "Implies --policy random.",
    )
    p.add_argument(
        "--base_model",
        default=None,
        help="Base model path/name for --policy vllm (shared by both role policies).",
    )
    p.add_argument(
        "--planner_lora",
        default=None,
        help="LoRA adapter path for the Planner role (--policy vllm; int_id 41). "
        "Optional (None = base model for the planner).",
    )
    p.add_argument(
        "--executor_lora",
        default=None,
        help="LoRA adapter path for the Executor role (--policy vllm; int_id 42). "
        "Optional (None = base model for the executor).",
    )
    p.add_argument(
        "--planner_model",
        default=None,
        help="FULL-PARAM mode: full model checkpoint for the Planner (its OWN vLLM engine, no LoRA). "
        "When set with --executor_model, two separate engines replace the shared-base+LoRA path.",
    )
    p.add_argument(
        "--executor_model",
        default=None,
        help="FULL-PARAM mode: full model checkpoint for the Executor (its OWN vLLM engine, no LoRA).",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Split passed to env.list_tasks when generating tasks without --tasks_jsonl.",
    )
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed (reproducible runs).")
    # vLLM sampling / capacity knobs (only used by --policy vllm).
    p.add_argument("--max_tokens", type=int, default=512, help="vLLM max new tokens per act.")
    p.add_argument("--temperature", type=float, default=0.8, help="vLLM sampling temperature.")
    p.add_argument("--top_p", type=float, default=0.95, help="vLLM nucleus top_p.")
    p.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.85,
        help="vLLM GPU memory utilization (only --policy vllm).",
    )
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Env construction                                                              #
# --------------------------------------------------------------------------- #
def build_env(args: argparse.Namespace):
    """Construct the requested AgenticEnv.

    Imported lazily so the mock path never touches the DeepPlanning module surface.
    Neither env imports torch/vllm at module top (dry-run constraint).
    """
    if args.env == "textcraft":
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
# Task acquisition                                                              #
# --------------------------------------------------------------------------- #
def load_tasks(args: argparse.Namespace, env) -> List[Task]:
    """Return the list of Task objects to roll out.

    Priority:
      1. --tasks_jsonl present -> read Task.from_dict per line (challenger output).
      2. else -> env.list_tasks(split); if that yields too few, top up via
         env.sample_new_task(rng, difficulty) until num_tasks is reached.
    --num_tasks caps the count (<=0 with --tasks_jsonl means 'use all provided').
    """
    rng = random.Random(args.seed)

    if args.tasks_jsonl:
        tasks: List[Task] = []
        path = Path(args.tasks_jsonl)
        if not path.is_file():
            raise FileNotFoundError(f"--tasks_jsonl not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                # Challenger may serialize the Task directly or wrap it under "task".
                if "task_id" not in d and isinstance(d.get("task"), dict):
                    d = d["task"]
                tasks.append(Task.from_dict(d))
        if args.num_tasks and args.num_tasks > 0:
            tasks = tasks[: args.num_tasks]
        if not tasks:
            raise ValueError(f"--tasks_jsonl {path} contained no tasks.")
        return tasks

    # No tasks file: pull from the env's split, then top up by sampling NEW tasks.
    want = args.num_tasks if args.num_tasks and args.num_tasks > 0 else 4
    tasks = list(env.list_tasks(args.split))
    if len(tasks) > want:
        tasks = tasks[:want]
    k = 0
    while len(tasks) < want:
        # Spread difficulty across [0,1] for the topped-up tasks.
        difficulty = (k + 1) / float(want + 1)
        tasks.append(env.sample_new_task(rng, difficulty))
        k += 1
    return tasks


# --------------------------------------------------------------------------- #
# Policy construction                                                           #
# --------------------------------------------------------------------------- #
def build_policies(args: argparse.Namespace, env):
    """Return (planner_policy, executor_policy).

    RandomPolicy path (dry_run / --policy random): NO torch, NO vllm. Two independent
    RandomPolicy instances (different seeds) so planner/executor decisions decorrelate;
    run_episode re-binds the live task onto each before every episode.

    vLLM path (--policy vllm): build ONE shared vllm.LLM (inside the first VLLMPolicy)
    and reuse it for the second role so we don't load the base model twice. vllm is
    lazy-imported inside VLLMPolicy only.
    """
    use_random = args.dry_run or args.policy == "random"

    if use_random:
        planner_policy = RandomPolicy(env, task=None, seed=args.seed + 41)
        executor_policy = RandomPolicy(env, task=None, seed=args.seed + 42)
        return planner_policy, executor_policy

    # --policy vllm

    # FULL-PARAM mode: planner/executor are two independent full checkpoints -> two SEPARATE
    # engines (no shared base, no LoRA). Both live on the one visible card, so each takes
    # ~half the memory budget (the caller passes a halved --gpu_memory_utilization).
    if args.planner_model and args.executor_model:
        planner_policy = VLLMPolicy(
            role="planner",
            lora=None,
            model=args.planner_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_lora=False,
        )
        executor_policy = VLLMPolicy(
            role="executor",
            lora=None,
            model=args.executor_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_lora=False,
        )
        return planner_policy, executor_policy

    if not args.base_model:
        raise ValueError("--policy vllm requires --base_model (or --planner_model/--executor_model for full-param).")

    # Build the planner policy first (it constructs the shared LLM), then hand the same
    # llm + tokenizer to the executor policy with its own LoRA + role int_id.
    planner_policy = VLLMPolicy(
        role="planner",
        lora=args.planner_lora,
        model=args.base_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    executor_policy = VLLMPolicy(
        role="executor",
        lora=args.executor_lora,
        llm=planner_policy.llm,
        tokenizer=getattr(planner_policy, "tokenizer", None),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    return planner_policy, executor_policy


# --------------------------------------------------------------------------- #
# Row assembly                                                                  #
# --------------------------------------------------------------------------- #
def trajectory_to_row(traj) -> Dict[str, Any]:
    """Project a Trajectory to the FROZEN JSONL row.

    Guarantees the five required keys are present:
        {task_id, sample_index, terminal_reward, turns, predicate_library}
    and passes through the Trajectory's extra keys (stage_records_global, info), which
    split_agentic_rollouts_by_role.py does not read. We build from traj.to_dict() so the
    Turn / StageRecord serialization is exactly the loop's Turn shape.
    """
    d = traj.to_dict()
    # Defensive: ensure the five required keys exist even if a future Trajectory drops
    # one (they are all present in the current rollout_agentic.Trajectory.to_dict()).
    row = {
        "task_id": d["task_id"],
        "sample_index": int(d.get("sample_index", 0)),
        "terminal_reward": float(d.get("terminal_reward", 0.0)),
        "turns": d.get("turns", []),
        "predicate_library": d.get("predicate_library", []),
    }
    # Pass through extras (not read by the role-split, useful for audits).
    if "stage_records_global" in d:
        row["stage_records_global"] = d["stage_records_global"]
    if "info" in d:
        row["info"] = d["info"]
    return row


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.dry_run and args.policy == "vllm":
        print("[build_agentic_rollouts] --dry_run forces RandomPolicy; ignoring --policy vllm.")
        args.policy = "random"

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = build_env(args)
    tasks = load_tasks(args, env)
    planner_policy, executor_policy = build_policies(args, env)

    print(f"[build_agentic_rollouts] env={args.env} policy={args.policy} "
          f"dry_run={args.dry_run}")
    print(f"[build_agentic_rollouts] n_tasks={len(tasks)} group_size={args.group_size} "
          f"max_steps={args.max_steps} max_replans={args.max_replans}")
    print(f"[build_agentic_rollouts] out={out_path}")

    n_rows = 0
    n_planner_turns = 0
    n_executor_turns = 0
    reward_sum = 0.0

    with out_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            for sample_index in range(max(1, int(args.group_size))):
                traj = run_episode(
                    env=env,
                    task=task,
                    planner=Planner,
                    executor=Executor,
                    planner_policy=planner_policy,
                    executor_policy=executor_policy,
                    max_steps=args.max_steps,
                    max_replans=args.max_replans,
                    sample_index=sample_index,
                )
                row = trajectory_to_row(traj)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
                reward_sum += float(row["terminal_reward"])
                for t in row["turns"]:
                    if t.get("role") == "planner":
                        n_planner_turns += 1
                    elif t.get("role") == "executor":
                        n_executor_turns += 1

    summary = {
        "out_jsonl": str(out_path),
        "env": args.env,
        "policy": args.policy,
        "dry_run": bool(args.dry_run),
        "n_tasks": len(tasks),
        "group_size": int(args.group_size),
        "n_rows": n_rows,
        "n_planner_turns": n_planner_turns,
        "n_executor_turns": n_executor_turns,
        "mean_terminal_reward": (reward_sum / n_rows) if n_rows else 0.0,
        "max_steps": int(args.max_steps),
        "max_replans": int(args.max_replans),
        "seed": int(args.seed),
    }
    summary_path = out_path.with_name("summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[build_agentic_rollouts] wrote {n_rows} rows "
          f"(planner_turns={n_planner_turns}, executor_turns={n_executor_turns}, "
          f"mean_terminal_reward={summary['mean_terminal_reward']:.4f})")
    print(f"[build_agentic_rollouts] summary -> {summary_path}")


if __name__ == "__main__":
    main()
    # Same JVM exit-hang guard as generate_agentic_challenger_tasks.py: ScienceWorld/ALFWorld leave
    # non-daemon threads that block interpreter shutdown after all rollouts + summary.json are written,
    # freezing the self-play driver. Force a clean exit (outputs already on disk).
    # CRITICAL companion: kill our direct children (vLLM EngineCore / java) BEFORE _exit -- plain
    # _exit orphans them holding ~40GB GPU each for minutes, OOM-ing the next stage's engine init.
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
    _os._exit(0)
