#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate trainable challenger candidates by sampling a challenger policy.

This is the TRUE trainable Challenger rollout stage.

It is different from generate_agentic_challenger_tasks.py:
  - generate_agentic_challenger_tasks.py samples tasks from env.sample_new_task(...)
    and gates them.
  - this script asks a challenger MODEL/POLICY to output <task>{Task JSON}</task>,
    then probes the produced task with the current solver/PATSP policy and records
    a reward-bearing candidate.

Output:
  candidates_jsonl:
    one row per challenger sample, including:
      prompt, completion, parsed task, format_ok, well_posed,
      probe_route, probe_score, probe_rewards

Optional:
  out_tasks_jsonl:
    valid in-band tasks usable for solver/planner/executor training.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.base import AgenticEnv, Task  # noqa: E402
from roles import Planner, Executor, Solver  # noqa: E402
from rollout_agentic import (  # noqa: E402
    RandomPolicy,
    VLLMPolicy,
    run_episode,
    run_solver_episode,
)
from build_agentic_rollouts import build_env  # noqa: E402


TASK_OPEN = "<task>"
TASK_CLOSE = "</task>"
TASK_BLOCK_RE = re.compile(
    re.escape(TASK_OPEN) + r"(?P<inner>.*?)" + re.escape(TASK_CLOSE),
    re.DOTALL,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample trainable challenger candidates and probe them."
    )

    p.add_argument(
        "--route",
        choices=["rzero", "patsp"],
        required=True,
        help="Which policy probes challenger-produced tasks.",
    )
    p.add_argument(
        "--env",
        choices=["textcraft", "scienceworld", "alfworld", "herobench", "longreason"],
        default="textcraft",
    )

    p.add_argument("--out_candidates_jsonl", required=True)
    p.add_argument(
        "--out_tasks_jsonl",
        default=None,
        help="Optional output of valid in-band Task JSONL for downstream solver/PATSP training.",
    )
    p.add_argument("--summary_json", default=None)

    p.add_argument("--num_prompts", type=int, default=1)
    p.add_argument("--samples_per_prompt", type=int, default=4)
    p.add_argument("--difficulty", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--policy",
        choices=["random", "vllm"],
        default="random",
        help="Challenger generation policy. random is CPU smoke stand-in.",
    )
    p.add_argument("--dry_run", action="store_true")

    # Challenger policy.
    p.add_argument("--base_model", default=os.environ.get("BASE_MODEL"))
    p.add_argument("--challenger_lora", default=None)
    p.add_argument("--challenger_model", default=None)

    # Probe policy for route=rzero.
    p.add_argument("--solver_lora", default=None)
    p.add_argument("--solver_model", default=None)

    # Probe policies for route=patsp.
    p.add_argument("--planner_lora", default=None)
    p.add_argument("--executor_lora", default=None)
    p.add_argument("--planner_model", default=None)
    p.add_argument("--executor_model", default=None)

    # Generation/probe controls.
    p.add_argument("--probe_k", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=8)
    p.add_argument("--max_replans", type=int, default=2)
    p.add_argument("--score_min", type=float, default=0.0)
    p.add_argument("--score_max", type=float, default=1.0)
    p.add_argument(
        "--probe_score_metric",
        choices=["mean_reward", "solve_rate"],
        default="mean_reward",
    )

    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)

    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print candidate parse/probe details.",
    )

    return p.parse_args(argv)


def task_to_dict(task: Task) -> Dict[str, Any]:
    if hasattr(task, "to_dict"):
        return task.to_dict()
    return {
        "task_id": task.task_id,
        "spec": task.spec,
        "constraints": task.constraints,
        "info": task.info,
    }


def task_from_dict(obj: Dict[str, Any]) -> Task:
    if hasattr(Task, "from_dict"):
        return Task.from_dict(obj)
    return Task(
        task_id=str(obj.get("task_id", "")),
        spec=obj.get("spec", {}) or {},
        constraints=obj.get("constraints", {}) or {},
        info=obj.get("info", {}) or {},
    )


def json_dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def build_reference_task(env: AgenticEnv, args: argparse.Namespace, rng: random.Random) -> Dict[str, Any]:
    """Sample one env-native task only as a schema/example for prompting."""
    try:
        t = env.sample_new_task(rng, args.difficulty)
        return task_to_dict(t)
    except Exception:
        return {
            "task_id": "example_task_id",
            "spec": {"text": "Describe the task here."},
            "constraints": {},
            "info": {"difficulty": args.difficulty},
        }


def build_challenger_messages(
    env: AgenticEnv,
    args: argparse.Namespace,
    prompt_index: int,
    reference_task: Dict[str, Any],
) -> List[Dict[str, str]]:
    system = (
        "You are the CHALLENGER in a self-improving verifier-grounded training system.\n"
        "Your job is to generate one new machine-checkable task that is neither trivial "
        "nor impossible for the current solver.\n"
        "Output exactly one task wrapped in <task>...</task> and nothing else.\n"
        "The content inside <task> must be a JSON object compatible with the Task schema:\n"
        '{"task_id": str, "spec": object, "constraints": object, "info": object}\n'
        "Do not output explanations, plans, markdown, or extra text."
    )

    user = "\n".join(
        [
            f"ENVIRONMENT: {args.env}",
            f"ROUTE TO PROBE: {args.route}",
            f"TARGET DIFFICULTY: {args.difficulty:.3f}",
            "",
            "REFERENCE TASK JSON SCHEMA/EXAMPLE:",
            json.dumps(reference_task, ensure_ascii=False, indent=2),
            "",
            "Generate a fresh task in the same schema family.",
            "The task must be solvable, verifiable, and useful for training.",
            "Return exactly:",
            "<task>{...valid Task JSON...}</task>",
            "",
            f"PROMPT_INDEX: {prompt_index}",
        ]
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    return "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)


class RandomChallengerPolicy:
    """CPU smoke stand-in for trainable challenger.

    It intentionally emits <task>{Task JSON}</task> so the rest of the trainable
    challenger pipeline can be tested without loading a model.
    """

    def __init__(self, env: AgenticEnv, difficulty: float, seed: int):
        self.env = env
        self.difficulty = float(difficulty)
        self.rng = random.Random(seed)
        self.counter = 0

    def act(self, messages: List[Dict[str, str]]) -> str:
        self.counter += 1
        task = self.env.sample_new_task(self.rng, self.difficulty)
        task.task_id = f"{task.task_id}-challenger-smoke-{self.counter:05d}"
        return TASK_OPEN + json_dumps_compact(task_to_dict(task)) + TASK_CLOSE


def build_challenger_policy(args: argparse.Namespace, env: AgenticEnv):
    if args.policy == "random" or args.dry_run:
        return RandomChallengerPolicy(env=env, difficulty=args.difficulty, seed=args.seed + 4400)

    if args.policy != "vllm":
        raise ValueError(f"unknown --policy {args.policy!r}")

    if args.challenger_model:
        return VLLMPolicy(
            role="challenger",
            lora=None,
            model=args.challenger_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_lora=False,
        )

    if not args.base_model:
        raise ValueError("--policy vllm requires --base_model or --challenger_model")

    use_lora = bool(args.challenger_lora)
    return VLLMPolicy(
        role="challenger",
        lora=args.challenger_lora if use_lora else None,
        model=args.base_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=use_lora,
    )


def parse_task_completion(text: str) -> Tuple[bool, Optional[Task], Optional[Dict[str, Any]], str]:
    full = str(text or "")
    m = TASK_BLOCK_RE.search(full)
    if not m:
        return False, None, None, "missing_task_block"

    inner = m.group("inner").strip()
    if not inner:
        return False, None, None, "empty_task_block"

    try:
        obj = json.loads(inner)
    except Exception as e:
        return False, None, None, f"json_parse_error:{type(e).__name__}:{e}"

    if not isinstance(obj, dict):
        return False, None, None, "task_json_not_object"

    try:
        task = task_from_dict(obj)
    except Exception as e:
        return False, None, obj, f"task_from_dict_error:{type(e).__name__}:{e}"

    if not getattr(task, "task_id", ""):
        task.task_id = "challenger_generated_task"

    task.info = dict(task.info or {})
    task.info.setdefault("source", "model_challenger")
    return True, task, obj, ""


def safe_well_posed(env: AgenticEnv, task: Task) -> Tuple[bool, str]:
    try:
        ok = bool(env.is_well_posed(task))
        return ok, "" if ok else "env_is_well_posed_false"
    except Exception as e:
        return False, f"env_is_well_posed_error:{type(e).__name__}:{e}"


def build_rzero_probe_policy(args: argparse.Namespace, env: AgenticEnv, seed: int):
    if args.policy == "random" or args.dry_run:
        return RandomPolicy(env, task=None, seed=5000 + seed)

    if args.solver_model:
        return VLLMPolicy(
            role="solver",
            lora=None,
            model=args.solver_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_lora=False,
        )

    if not args.base_model:
        raise ValueError("--route rzero --policy vllm requires --base_model or --solver_model")

    use_lora = bool(args.solver_lora)
    return VLLMPolicy(
        role="solver",
        lora=args.solver_lora if use_lora else None,
        model=args.base_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=use_lora,
    )



def _same_model_ref(a, b) -> bool:
    if not a or not b:
        return False
    try:
        return Path(str(a)).resolve() == Path(str(b)).resolve()
    except Exception:
        return str(a) == str(b)


def _cleanup_policy_memory(obj=None) -> None:
    """Best-effort cleanup before loading a second vLLM engine.

    This is important for PATSP challenger probing: generation may use a
    challenger engine, then probing needs planner/executor engines. On a 24GB
    GPU, stale vLLM references can make the next engine fail at startup.
    """
    try:
        if obj is not None:
            del obj
    except Exception:
        pass

    try:
        import gc
        gc.collect()
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def build_patsp_probe_policies(args: argparse.Namespace, env: AgenticEnv, seed: int):
    """Build PATSP planner/executor probe policies.

    PLAN-AND-ACT requires two roles semantically, but it does not require two
    separate vLLM engines when planner/executor share the same model. Sharing
    is critical for 24GB GPUs: challenger probing often otherwise tries to keep
    challenger + planner + executor engines on GPU 0.
    """
    if args.policy == "random" or args.dry_run:
        return (
            RandomPolicy(env, task=None, seed=seed + 41),
            RandomPolicy(env, task=None, seed=seed + 42),
        )

    if args.policy != "vllm":
        raise ValueError(f"Unsupported policy for PATSP probe: {args.policy}")

    # Full-param mode: if planner and executor are the same checkpoint, use one
    # shared vLLM engine with different role int_ids.
    if args.planner_model and args.executor_model:
        if _same_model_ref(args.planner_model, args.executor_model):
            planner_policy = VLLMPolicy(
                role="planner",
                lora=None,
                model=args.planner_model,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            executor_policy = VLLMPolicy(
                role="executor",
                lora=None,
                llm=planner_policy.llm,
                tokenizer=getattr(planner_policy, "tokenizer", None),
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            return planner_policy, executor_policy

        # True separate full checkpoints. This may require more GPU engineering
        # in later multi-round full-param PATSP, but keep the path available.
        planner_policy = VLLMPolicy(
            role="planner",
            lora=None,
            model=args.planner_model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        executor_policy = VLLMPolicy(
            role="executor",
            lora=None,
            model=args.executor_model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        return planner_policy, executor_policy

    if not args.base_model:
        raise ValueError("--route patsp --policy vllm requires --base_model or planner/executor models")

    # Shared-base path. This is the normal one-round cold-start PATSP smoke:
    # planner and executor are semantically separate, but share one base engine.
    planner_lora = args.planner_lora or None
    executor_lora = args.executor_lora or None

    planner_policy = VLLMPolicy(
        role="planner",
        lora=planner_lora,
        model=args.base_model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    executor_policy = VLLMPolicy(
        role="executor",
        lora=executor_lora,
        llm=planner_policy.llm,
        tokenizer=getattr(planner_policy, "tokenizer", None),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    return planner_policy, executor_policy


def probe_task(env: AgenticEnv, task: Task, args: argparse.Namespace, base_seed: int) -> Dict[str, Any]:
    rewards: List[float] = []
    solves = 0

    for k in range(max(1, int(args.probe_k))):
        seed = base_seed * 1000 + k

        if args.route == "rzero":
            solver_policy = build_rzero_probe_policy(args, env, seed)
            traj = run_solver_episode(
                env=env,
                task=task,
                solver=Solver,
                solver_policy=solver_policy,
                max_steps=args.max_steps,
                sample_index=k,
            )
        else:
            planner_policy, executor_policy = build_patsp_probe_policies(args, env, seed)
            traj = run_episode(
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
        if r >= 1.0 - 1e-9:
            solves += 1

    solve_rate = solves / float(max(1, int(args.probe_k)))
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    score = mean_reward if args.probe_score_metric == "mean_reward" else solve_rate

    return {
        "probe_route": args.route,
        "probe_score": float(score),
        "probe_score_metric": args.probe_score_metric,
        "probe_rewards": rewards,
        "probe_mean_reward": float(mean_reward),
        "probe_solve_rate": float(solve_rate),
        "probe_k": int(max(1, int(args.probe_k))),
    }


def in_band(score: float, args: argparse.Namespace) -> bool:
    return float(args.score_min) <= float(score) <= float(args.score_max)



def cleanup_policy_memory(obj: Any = None) -> None:
    """Best-effort cleanup before loading a second full model."""
    try:
        if obj is not None:
            del obj
    except Exception:
        pass

    try:
        import gc
        gc.collect()
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def default_probe_payload(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "probe_route": args.route,
        "probe_score": 0.0,
        "probe_score_metric": args.probe_score_metric,
        "probe_rewards": [],
        "probe_mean_reward": 0.0,
        "probe_solve_rate": 0.0,
        "probe_k": int(max(1, int(args.probe_k))),
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    out_candidates = Path(args.out_candidates_jsonl)
    out_candidates.parent.mkdir(parents=True, exist_ok=True)

    out_tasks = Path(args.out_tasks_jsonl) if args.out_tasks_jsonl else None
    if out_tasks:
        out_tasks.parent.mkdir(parents=True, exist_ok=True)

    summary_path = (
        Path(args.summary_json)
        if args.summary_json
        else out_candidates.with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    env = build_env(args)
    reference_task = build_reference_task(env, args, rng)

    # ------------------------------------------------------------------ #
    # Phase 1: challenger model only generates candidate task completions.
    # ------------------------------------------------------------------ #
    challenger_policy = build_challenger_policy(args, env)

    rows: List[Dict[str, Any]] = []
    n_candidates = 0
    n_format_ok = 0
    n_well_posed = 0
    bad_reasons: Dict[str, int] = {}

    for prompt_index in range(max(1, int(args.num_prompts))):
        messages = build_challenger_messages(
            env=env,
            args=args,
            prompt_index=prompt_index,
            reference_task=reference_task,
        )
        prompt = messages_to_prompt(messages)
        uid = f"{args.env}_{args.route}_challenger_prompt_{prompt_index:05d}"

        for sample_index in range(max(1, int(args.samples_per_prompt))):
            completion = challenger_policy.act(messages)
            n_candidates += 1

            format_ok, task, task_obj, parse_reason = parse_task_completion(completion)
            if format_ok:
                n_format_ok += 1
            else:
                bad_reasons[parse_reason] = bad_reasons.get(parse_reason, 0) + 1

            well_posed = False
            well_reason = "not_parsed"
            task_for_row: Any = task_obj
            task_for_probe: Optional[Dict[str, Any]] = None

            if task is not None:
                task.task_id = f"{task.task_id}-p{prompt_index:05d}-s{sample_index:05d}"
                task.info = dict(task.info or {})
                task.info["challenger_uid"] = uid
                task.info["challenger_sample_index"] = int(sample_index)
                task.info["challenger_route"] = args.route

                well_posed, well_reason = safe_well_posed(env, task)
                if well_posed:
                    n_well_posed += 1
                    task_for_probe = task_to_dict(task)
                    task_for_row = task_for_probe
                else:
                    bad_reasons[well_reason] = bad_reasons.get(well_reason, 0) + 1

            row = {
                "uid": uid,
                "sample_index": int(sample_index),
                "prompt_index": int(prompt_index),
                "prompt": prompt,
                "completion": completion,
                "format_ok": bool(format_ok),
                "parse_reason": parse_reason,
                "well_posed": bool(well_posed),
                "well_posed_reason": well_reason,
                "task": task_for_row,
                "env": args.env,
                "route": args.route,
                "difficulty": float(args.difficulty),
                "score_min": float(args.score_min),
                "score_max": float(args.score_max),
                "_task_for_probe": task_for_probe,
                **default_probe_payload(args),
            }
            rows.append(row)

    # Important for full-param mode:
    # release challenger engine before loading solver engine for probing.
    cleanup_policy_memory(challenger_policy)

    # ------------------------------------------------------------------ #
    # Release challenger generation engine before loading probe policy.
    # This prevents PATSP from keeping challenger+planner+executor engines
    # simultaneously on a 24GB GPU.
    try:
        _cleanup_policy_memory(challenger_policy)
        challenger_policy = None
    except NameError:
        _cleanup_policy_memory()


    # Phase 2: probe parsed/well-posed tasks with current solver/PATSP.
    # ------------------------------------------------------------------ #
    n_probed = 0
    n_in_band = 0
    n_valid_tasks_written = 0
    valid_tasks: List[Dict[str, Any]] = []

    for row in rows:
        task_dict = row.get("_task_for_probe")

        if task_dict is not None:
            try:
                task = task_from_dict(task_dict)
                probe = probe_task(
                    env=env,
                    task=task,
                    args=args,
                    base_seed=args.seed + int(row["prompt_index"]) * 10000 + int(row["sample_index"]),
                )
                row.update(probe)
                n_probed += 1

                if in_band(float(row["probe_score"]), args):
                    n_in_band += 1
                    valid_tasks.append(task_dict)
                    n_valid_tasks_written += 1
            except Exception as e:
                reason = f"probe_error:{type(e).__name__}:{e}"
                row["probe_error"] = reason
                bad_reasons[reason] = bad_reasons.get(reason, 0) + 1

        row.pop("_task_for_probe", None)

        if args.verbose:
            print(
                f"[candidate] uid={row['uid']} sample={row['sample_index']} "
                f"format_ok={row['format_ok']} well_posed={row['well_posed']} "
                f"score={float(row.get('probe_score', 0.0)):.4f}"
            )

    with out_candidates.open("w", encoding="utf-8") as cand_f:
        for row in rows:
            cand_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if out_tasks is not None:
        with out_tasks.open("w", encoding="utf-8") as task_f:
            for task_dict in valid_tasks:
                task_f.write(json.dumps(task_dict, ensure_ascii=False) + "\n")

    summary = {
        "out_candidates_jsonl": str(out_candidates),
        "out_tasks_jsonl": str(out_tasks) if out_tasks else None,
        "route": args.route,
        "env": args.env,
        "policy": args.policy,
        "dry_run": bool(args.dry_run),
        "num_prompts": int(args.num_prompts),
        "samples_per_prompt": int(args.samples_per_prompt),
        "n_candidates": n_candidates,
        "n_format_ok": n_format_ok,
        "n_well_posed": n_well_posed,
        "n_probed": n_probed,
        "n_in_band": n_in_band,
        "n_valid_tasks_written": n_valid_tasks_written,
        "bad_reasons": bad_reasons,
        "probe_score_metric": args.probe_score_metric,
        "score_min": float(args.score_min),
        "score_max": float(args.score_max),
        "two_phase_generation_then_probe": True,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[build_agentic_challenger_rollouts]", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
