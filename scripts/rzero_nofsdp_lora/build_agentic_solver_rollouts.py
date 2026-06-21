#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monolithic R-Zero solver rollout driver.

This script is the R-Zero counterpart of build_agentic_rollouts.py.

It emits agentic trajectories with ONLY role="solver" turns:

    {task_id, sample_index, terminal_reward, turns:[SolverTurn...], predicate_library}

Conceptual contract:
  - R-Zero has one solver model.
  - The solver does not receive a top-level plan.
  - The solver directly maps task + observation + available actions + history
    to one concrete environment action per step.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional, List, Any, Dict

# Robust import: sibling modules, no torch/vllm at module import time.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roles import Solver  # noqa: E402
from rollout_agentic import RandomPolicy, VLLMPolicy, run_solver_episode  # noqa: E402
from build_agentic_rollouts import build_env, load_tasks  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run monolithic R-Zero solver rollouts -> solver trajectories JSONL.",
    )
    p.add_argument(
        "--env",
        choices=["textcraft", "scienceworld", "alfworld", "herobench", "longreason"],
        default="textcraft",
        help="Environment benchmark.",
    )
    p.add_argument(
        "--group_size",
        type=int,
        default=4,
        help="Episodes per task; shared uid=task_id for group-relative advantage later.",
    )
    p.add_argument(
        "--num_tasks",
        type=int,
        default=4,
        help="Number of tasks to roll out. With --tasks_jsonl, caps provided tasks; <=0 means all.",
    )
    p.add_argument(
        "--tasks_jsonl",
        default=None,
        help="Optional challenger/task JSONL. If absent, tasks come from env.list_tasks(split).",
    )
    p.add_argument(
        "--max_steps",
        type=int,
        default=16,
        help="Max solver actions per episode.",
    )
    p.add_argument(
        "--out_jsonl",
        required=True,
        help="Output trajectory JSONL.",
    )
    p.add_argument(
        "--policy",
        choices=["vllm", "random"],
        default="random",
        help="'random' = CPU RandomPolicy; 'vllm' = VLLMPolicy(role='solver').",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Force RandomPolicy; no vLLM.",
    )
    p.add_argument(
        "--base_model",
        default=None,
        help="Base model path/name for --policy vllm LoRA mode.",
    )
    p.add_argument(
        "--solver_lora",
        default=None,
        help="LoRA adapter path for the solver role in shared-base mode.",
    )
    p.add_argument(
        "--solver_model",
        default=None,
        help="FULL-PARAM mode: full solver checkpoint path, no LoRA.",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Split passed to env.list_tasks when --tasks_jsonl is absent.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.85,
        help="vLLM GPU memory utilization.",
    )
    return p.parse_args(argv)


def build_solver_policy(args: argparse.Namespace, env):
    use_random = bool(args.dry_run) or args.policy == "random"

    if use_random:
        return RandomPolicy(env, task=None, seed=args.seed + 43)

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
        raise ValueError("--policy vllm requires --base_model or --solver_model")

    return VLLMPolicy(
        role="solver",
        lora=args.solver_lora,
        model=args.base_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
    )


def trajectory_to_row(traj) -> Dict[str, Any]:
    d = traj.to_dict()
    row = {
        "task_id": d["task_id"],
        "sample_index": int(d.get("sample_index", 0)),
        "terminal_reward": float(d.get("terminal_reward", 0.0)),
        "turns": d.get("turns", []),
        "predicate_library": d.get("predicate_library", []),
    }
    if "stage_records_global" in d:
        row["stage_records_global"] = d["stage_records_global"]
    if "info" in d:
        row["info"] = d["info"]
    return row


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.dry_run and args.policy == "vllm":
        print("[build_agentic_solver_rollouts] --dry_run forces RandomPolicy; ignoring --policy vllm.")
        args.policy = "random"

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    env = build_env(args)
    tasks = load_tasks(args, env)
    solver_policy = build_solver_policy(args, env)

    print(
        f"[build_agentic_solver_rollouts] env={args.env} policy={args.policy} "
        f"dry_run={bool(args.dry_run)}"
    )
    print(
        f"[build_agentic_solver_rollouts] n_tasks={len(tasks)} group_size={args.group_size} "
        f"max_steps={args.max_steps}"
    )
    print(f"[build_agentic_solver_rollouts] out={out_path}")

    n_rows = 0
    n_solver_turns = 0
    n_non_solver_turns = 0
    reward_sum = 0.0

    with out_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            for sample_index in range(max(1, int(args.group_size))):
                traj = run_solver_episode(
                    env=env,
                    task=task,
                    solver=Solver,
                    solver_policy=solver_policy,
                    max_steps=args.max_steps,
                    sample_index=sample_index,
                )

                row = trajectory_to_row(traj)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

                n_rows += 1
                reward_sum += float(row["terminal_reward"])

                for turn in row["turns"]:
                    if turn.get("role") == "solver":
                        n_solver_turns += 1
                    else:
                        n_non_solver_turns += 1

    summary = {
        "out_jsonl": str(out_path),
        "env": args.env,
        "policy": args.policy,
        "dry_run": bool(args.dry_run),
        "n_tasks": len(tasks),
        "group_size": int(args.group_size),
        "n_rows": n_rows,
        "n_solver_turns": n_solver_turns,
        "n_non_solver_turns": n_non_solver_turns,
        "mean_terminal_reward": (reward_sum / n_rows) if n_rows else 0.0,
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "route": "rzero_monolithic_solver",
    }

    summary_path = out_path.with_name("summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[build_agentic_solver_rollouts] wrote {n_rows} rows "
        f"(solver_turns={n_solver_turns}, non_solver_turns={n_non_solver_turns}, "
        f"mean_terminal_reward={summary['mean_terminal_reward']:.4f})"
    )
    print(f"[build_agentic_solver_rollouts] summary -> {summary_path}")


if __name__ == "__main__":
    main()

    # Same child-process exit guard pattern as build_agentic_rollouts.py.
    import os as _os, sys as _sys, glob as _glob, signal as _signal

    _me = _os.getpid()
    for _stat in _glob.glob("/proc/[0-9]*/stat"):
        try:
            _data = open(_stat).read()
            _after = _data[_data.rindex(")") + 2:].split()
            if int(_after[1]) == _me:
                _os.kill(int(_stat.split("/")[2]), _signal.SIGKILL)
        except Exception:
            pass

    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(0)