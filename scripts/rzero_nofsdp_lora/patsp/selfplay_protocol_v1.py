#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protocol v1 driver for R-Zero vs PATSP experiments.

Initial implementation:
  - route=rzero only
  - monolithic solver rollout only
  - build strict solver rows
  - optional training is intentionally not implemented yet

This file exists to lock the correct protocol shape before adding training:

R-Zero:
  challenger + solver

PATSP:
  challenger + planner + executor

Do NOT implement R-Zero as solver plugged into planner/executor slots.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "scripts" / "rzero_nofsdp_lora"


def run_cmd(cmd: List[str], *, env: Optional[Dict[str, str]] = None) -> None:
    print("\n[protocol] RUN:")
    print(" ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True, env=env)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Protocol v1 self-play driver.")

    p.add_argument(
        "--route",
        choices=["rzero"],
        required=True,
        help="Initial version supports only route=rzero.",
    )
    p.add_argument(
        "--env",
        choices=["textcraft", "scienceworld", "alfworld", "herobench", "longreason"],
        default="textcraft",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--num_tasks", type=int, default=1)
    p.add_argument("--group_size", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--tasks_jsonl",
        default=None,
        help="Optional fixed/challenger task JSONL. If omitted, env.list_tasks(split) is used.",
    )
    p.add_argument("--split", default="train")

    p.add_argument(
        "--policy",
        choices=["random", "vllm"],
        default="random",
        help="Use random for CPU smoke; vllm for real model rollout.",
    )
    p.add_argument("--dry_run", action="store_true")

    p.add_argument("--base_model", default=os.environ.get("BASE_MODEL"))
    p.add_argument("--solver_lora", default=None)
    p.add_argument("--solver_model", default=None)

    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)

    p.add_argument(
        "--skip_train",
        action="store_true",
        help="Required in this initial smoke driver. Training will be added after CLI audit.",
    )

    return p.parse_args(argv)


def json_load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def assert_solver_only_trajectories(path: Path) -> Dict[str, int]:
    n_traj = 0
    n_turn = 0
    bad_roles: Dict[str, int] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            n_traj += 1
            row = json.loads(line)
            for turn in row.get("turns", []) or []:
                n_turn += 1
                role = str(turn.get("role", ""))
                if role != "solver":
                    bad_roles[role] = bad_roles.get(role, 0) + 1

    if bad_roles:
        raise RuntimeError(
            f"R-Zero trajectory contains non-solver roles: {bad_roles}. "
            "This violates the monolithic solver protocol."
        )

    return {"n_trajectories": n_traj, "n_solver_turns": n_turn}


def assert_solver_rows(path: Path) -> Dict[str, int]:
    n_rows = 0
    bad_roles: Dict[str, int] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            n_rows += 1
            row = json.loads(line)
            role = str(row.get("role", ""))
            if role != "solver":
                bad_roles[role] = bad_roles.get(role, 0) + 1

    if bad_roles:
        raise RuntimeError(
            f"R-Zero solver rows contain non-solver roles: {bad_roles}."
        )

    return {"n_solver_rows": n_rows}


def run_rzero_round(args: argparse.Namespace, round_idx: int, solver_ref: str) -> Dict[str, Any]:
    out_dir = Path(args.out_dir).resolve()
    round_dir = out_dir / f"r{round_idx}"
    round_dir.mkdir(parents=True, exist_ok=True)

    traj_jsonl = round_dir / f"r{round_idx}_traj.jsonl"
    solver_jsonl = round_dir / f"r{round_idx}_solver.jsonl"
    solver_rows_summary = round_dir / f"r{round_idx}_solver_rows.summary.json"

    rollout_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_agentic_solver_rollouts.py"),
        "--env", args.env,
        "--group_size", str(args.group_size),
        "--num_tasks", str(args.num_tasks),
        "--max_steps", str(args.max_steps),
        "--out_jsonl", str(traj_jsonl),
        "--policy", args.policy,
        "--split", args.split,
        "--seed", str(args.seed + round_idx),
        "--max_tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--gpu_memory_utilization", str(args.gpu_memory_utilization),
    ]

    if args.dry_run:
        rollout_cmd.append("--dry_run")

    if args.tasks_jsonl:
        rollout_cmd += ["--tasks_jsonl", args.tasks_jsonl]

    if args.policy == "vllm":
        if args.solver_model:
            rollout_cmd += ["--solver_model", args.solver_model]
        else:
            if not args.base_model:
                raise ValueError("--policy vllm requires --base_model or --solver_model")
            rollout_cmd += ["--base_model", args.base_model]
            if args.solver_lora:
                rollout_cmd += ["--solver_lora", args.solver_lora]

    run_cmd(rollout_cmd)

    traj_stats = assert_solver_only_trajectories(traj_jsonl)

    rows_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_agentic_solver_rows.py"),
        "--trajectories_jsonl", str(traj_jsonl),
        "--out_solver", str(solver_jsonl),
        "--summary_json", str(solver_rows_summary),
        "--require_nonempty",
    ]
    run_cmd(rows_cmd)

    row_stats = assert_solver_rows(solver_jsonl)

    rollout_summary_path = round_dir / "summary.json"
    rollout_summary = json_load(rollout_summary_path) if rollout_summary_path.exists() else {}
    rows_summary = json_load(solver_rows_summary)

    if not args.skip_train:
        raise NotImplementedError(
            "Training is intentionally not implemented in this initial protocol driver. "
            "Run with --skip_train. We will add training after auditing trainer CLI."
        )

    return {
        "round": round_idx,
        "route": "rzero",
        "solver_in": solver_ref,
        "traj_jsonl": str(traj_jsonl),
        "solver_jsonl": str(solver_jsonl),
        "solver_out": solver_ref,
        "trained": False,
        "traj_stats": traj_stats,
        "row_stats": row_stats,
        "rollout_summary": rollout_summary,
        "rows_summary": rows_summary,
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.route != "rzero":
        raise ValueError("Initial protocol driver supports only --route rzero")

    if not args.skip_train:
        raise NotImplementedError(
            "Initial driver only validates rollout->rows protocol. Use --skip_train."
        )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rounds: List[Dict[str, Any]] = []

    solver_ref = args.solver_model or args.solver_lora or args.base_model or "random_policy"

    for r in range(1, int(args.rounds) + 1):
        rec = run_rzero_round(args, r, solver_ref)
        rounds.append(rec)

    rounds_path = out_dir / "rounds.json"
    with rounds_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "route": "rzero",
                "env": args.env,
                "rounds": rounds,
                "protocol": "monolithic_solver_smoke",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n[protocol] wrote {rounds_path}")
    print("[protocol] OK: rzero monolithic solver protocol smoke finished")


if __name__ == "__main__":
    main()
