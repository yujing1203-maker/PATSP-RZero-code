#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protocol v1 driver for R-Zero vs PATSP experiments.

Current implementation:
  - route=rzero only
  - trainable challenger rollout stage is included
  - monolithic solver rollout stage is included
  - training is intentionally still behind --skip_train

R-Zero round shape:

  challenger_{i-1}
      -> generate challenger candidates
      -> probe candidates with solver_{i-1}
      -> build challenger rows with reward/advantage
      -> [future] train challenger_i

  challenger_i / candidate tasks
      -> solver rollouts with solver_{i-1}
      -> build solver rows
      -> [future] train solver_i

This driver must never implement R-Zero as planner+executor.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "scripts" / "rzero_nofsdp_lora"


def run_cmd(cmd: List[str], *, log_path: Optional[Path] = None) -> None:
    print("\n[protocol] RUN:")
    print(" ".join(str(x) for x in cmd))

    if log_path is None:
        subprocess.run(cmd, check=True)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)

    if rc != 0:
        raise RuntimeError(
            f"Command failed rc={rc}. See log: {log_path}\n"
            f"Command prefix: {' '.join(str(x) for x in cmd[:8])}"
        )


def json_load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_jsonl(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Protocol v1 self-play driver.")

    p.add_argument("--route", choices=["rzero"], required=True)
    p.add_argument(
        "--env",
        choices=["textcraft", "scienceworld", "alfworld", "herobench", "longreason"],
        default="textcraft",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)

    # Challenger candidate generation.
    p.add_argument("--difficulty", type=float, default=0.5)
    p.add_argument("--challenger_num_prompts", type=int, default=1)
    p.add_argument("--challenger_samples_per_prompt", type=int, default=4)
    p.add_argument("--probe_k", type=int, default=2)
    p.add_argument("--score_min", type=float, default=0.0)
    p.add_argument("--score_max", type=float, default=1.0)
    p.add_argument(
        "--probe_score_metric",
        choices=["mean_reward", "solve_rate"],
        default="mean_reward",
    )

    # Solver rollout controls.
    p.add_argument("--group_size", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=4)
    p.add_argument("--max_replans", type=int, default=2)
    p.add_argument("--split", default="train")

    # Policy/model controls.
    p.add_argument("--policy", choices=["random", "vllm"], default="random")
    p.add_argument("--dry_run", action="store_true")

    p.add_argument("--base_model", default=os.environ.get("BASE_MODEL"))

    p.add_argument("--challenger_lora", default=None)
    p.add_argument("--challenger_model", default=None)

    p.add_argument("--solver_lora", default=None)
    p.add_argument("--solver_model", default=None)

    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)

    p.add_argument(
        "--skip_train",
        action="store_true",
        help="Required for this protocol smoke version. Training is added later.",
    )

    return p.parse_args(argv)


def assert_challenger_candidates(path: Path, route: str) -> Dict[str, int]:
    n = 0
    n_bad_route = 0
    n_missing_reward_inputs = 0
    n_format_ok = 0
    n_well_posed = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            row = json.loads(line)

            if row.get("probe_route") != route:
                n_bad_route += 1

            if "probe_score" not in row or "probe_rewards" not in row:
                n_missing_reward_inputs += 1

            if row.get("format_ok"):
                n_format_ok += 1
            if row.get("well_posed"):
                n_well_posed += 1

    if n == 0:
        raise RuntimeError(f"No challenger candidates emitted: {path}")
    if n_bad_route:
        raise RuntimeError(f"Found challenger candidates with wrong probe_route in {path}")
    if n_missing_reward_inputs:
        raise RuntimeError(f"Found challenger candidates missing probe_score/probe_rewards in {path}")

    return {
        "n_challenger_candidates": n,
        "n_format_ok": n_format_ok,
        "n_well_posed": n_well_posed,
    }


def assert_challenger_rows(path: Path) -> Dict[str, int]:
    n = 0
    bad_roles: Dict[str, int] = {}
    missing_signal = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            row = json.loads(line)

            role = str(row.get("role", ""))
            if role != "challenger":
                bad_roles[role] = bad_roles.get(role, 0) + 1

            if "reward" not in row or "advantage" not in row:
                missing_signal += 1

    if n == 0:
        raise RuntimeError(f"No challenger rows emitted: {path}")
    if bad_roles:
        raise RuntimeError(f"Challenger rows contain non-challenger roles: {bad_roles}")
    if missing_signal:
        raise RuntimeError(f"Challenger rows missing reward/advantage: {missing_signal}")

    return {"n_challenger_rows": n}


def assert_solver_only_trajectories(path: Path) -> Dict[str, int]:
    n_traj = 0
    n_turn = 0
    bad_roles: Dict[str, int] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            n_traj += 1
            row = json.loads(line)
            for turn in row.get("turns", []) or []:
                n_turn += 1
                role = str(turn.get("role", ""))
                if role != "solver":
                    bad_roles[role] = bad_roles.get(role, 0) + 1

    if n_traj == 0:
        raise RuntimeError(f"No solver trajectories emitted: {path}")
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
            if not line.strip():
                continue

            n_rows += 1
            row = json.loads(line)
            role = str(row.get("role", ""))
            if role != "solver":
                bad_roles[role] = bad_roles.get(role, 0) + 1

    if n_rows == 0:
        raise RuntimeError(f"No solver rows emitted: {path}")
    if bad_roles:
        raise RuntimeError(f"R-Zero solver rows contain non-solver roles: {bad_roles}.")

    return {"n_solver_rows": n_rows}


def append_vllm_common_args(cmd: List[str], args: argparse.Namespace) -> None:
    cmd += [
        "--max_tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--gpu_memory_utilization", str(args.gpu_memory_utilization),
    ]


def append_challenger_model_args(cmd: List[str], args: argparse.Namespace) -> None:
    if args.policy != "vllm":
        return

    if args.challenger_model:
        cmd += ["--challenger_model", args.challenger_model]
    else:
        if not args.base_model:
            raise ValueError("--policy vllm requires --base_model or --challenger_model")
        cmd += ["--base_model", args.base_model]
        if args.challenger_lora:
            cmd += ["--challenger_lora", args.challenger_lora]


def append_solver_probe_args(cmd: List[str], args: argparse.Namespace) -> None:
    if args.policy != "vllm":
        return

    if args.solver_model:
        cmd += ["--solver_model", args.solver_model]
    else:
        if not args.base_model:
            raise ValueError("--policy vllm requires --base_model or --solver_model")
        cmd += ["--base_model", args.base_model]
        if args.solver_lora:
            cmd += ["--solver_lora", args.solver_lora]


def run_rzero_round(
    args: argparse.Namespace,
    round_idx: int,
    challenger_ref: str,
    solver_ref: str,
) -> Dict[str, Any]:
    out_dir = Path(args.out_dir).resolve()
    round_dir = out_dir / f"r{round_idx}"
    round_dir.mkdir(parents=True, exist_ok=True)

    candidates_jsonl = round_dir / f"r{round_idx}_challenger_candidates.jsonl"
    challenger_rows_jsonl = round_dir / f"r{round_idx}_challenger.jsonl"
    challenger_rows_summary = round_dir / f"r{round_idx}_challenger_rows.summary.json"
    challenger_tasks_jsonl = round_dir / f"r{round_idx}_tasks.jsonl"
    challenger_rollout_summary = round_dir / f"r{round_idx}_challenger_rollouts.summary.json"

    traj_jsonl = round_dir / f"r{round_idx}_traj.jsonl"
    solver_jsonl = round_dir / f"r{round_idx}_solver.jsonl"
    solver_rows_summary = round_dir / f"r{round_idx}_solver_rows.summary.json"

    # 1. Challenger policy produces task candidates and receives probe results.
    challenger_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_agentic_challenger_rollouts.py"),
        "--route", "rzero",
        "--env", args.env,
        "--out_candidates_jsonl", str(candidates_jsonl),
        "--out_tasks_jsonl", str(challenger_tasks_jsonl),
        "--summary_json", str(challenger_rollout_summary),
        "--num_prompts", str(args.challenger_num_prompts),
        "--samples_per_prompt", str(args.challenger_samples_per_prompt),
        "--difficulty", str(args.difficulty),
        "--seed", str(args.seed + round_idx * 100),
        "--probe_k", str(args.probe_k),
        "--max_steps", str(args.max_steps),
        "--score_min", str(args.score_min),
        "--score_max", str(args.score_max),
        "--probe_score_metric", args.probe_score_metric,
        "--policy", args.policy,
    ]
    if args.dry_run:
        challenger_cmd.append("--dry_run")
    append_vllm_common_args(challenger_cmd, args)
    append_challenger_model_args(challenger_cmd, args)
    append_solver_probe_args(challenger_cmd, args)

    run_cmd(challenger_cmd, log_path=round_dir / f"r{round_idx}_challenger_rollouts.log")
    challenger_candidate_stats = assert_challenger_candidates(candidates_jsonl, "rzero")

    if not challenger_tasks_jsonl.is_file() or count_jsonl(challenger_tasks_jsonl) == 0:
        raise RuntimeError(
            f"No valid challenger-produced tasks written to {challenger_tasks_jsonl}. "
            "For smoke tests, use --score_min 0.0 --score_max 1.0 and inspect challenger rollout log."
        )

    # 2. Convert challenger candidates into trainable challenger rows.
    challenger_rows_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_agentic_challenger_rows.py"),
        "--candidates_jsonl", str(candidates_jsonl),
        "--out_challenger", str(challenger_rows_jsonl),
        "--summary_json", str(challenger_rows_summary),
        "--min_score", str(args.score_min),
        "--max_score", str(args.score_max),
        "--require_nonempty",
    ]
    run_cmd(challenger_rows_cmd, log_path=round_dir / f"r{round_idx}_challenger_rows.log")
    challenger_row_stats = assert_challenger_rows(challenger_rows_jsonl)

    if not args.skip_train:
        raise NotImplementedError(
            "Training is intentionally not implemented in this protocol version. "
            "Next step will add challenger train, then solver train."
        )

    # 3. Use challenger-produced tasks for solver rollouts.
    solver_rollout_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_agentic_solver_rollouts.py"),
        "--env", args.env,
        "--tasks_jsonl", str(challenger_tasks_jsonl),
        "--num_tasks", "0",
        "--group_size", str(args.group_size),
        "--max_steps", str(args.max_steps),
        "--out_jsonl", str(traj_jsonl),
        "--policy", args.policy,
        "--split", args.split,
        "--seed", str(args.seed + round_idx * 1000),
    ]
    if args.dry_run:
        solver_rollout_cmd.append("--dry_run")
    append_vllm_common_args(solver_rollout_cmd, args)

    if args.policy == "vllm":
        if args.solver_model:
            solver_rollout_cmd += ["--solver_model", args.solver_model]
        else:
            solver_rollout_cmd += ["--base_model", args.base_model]
            if args.solver_lora:
                solver_rollout_cmd += ["--solver_lora", args.solver_lora]

    run_cmd(solver_rollout_cmd, log_path=round_dir / f"r{round_idx}_solver_rollouts.log")
    traj_stats = assert_solver_only_trajectories(traj_jsonl)

    # 4. Convert solver trajectories into strict solver training rows.
    solver_rows_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_agentic_solver_rows.py"),
        "--trajectories_jsonl", str(traj_jsonl),
        "--out_solver", str(solver_jsonl),
        "--summary_json", str(solver_rows_summary),
        "--require_nonempty",
    ]
    run_cmd(solver_rows_cmd, log_path=round_dir / f"r{round_idx}_solver_rows.log")
    row_stats = assert_solver_rows(solver_jsonl)

    challenger_rollout_summary_obj = (
        json_load(challenger_rollout_summary)
        if challenger_rollout_summary.exists()
        else {}
    )
    challenger_rows_summary_obj = (
        json_load(challenger_rows_summary)
        if challenger_rows_summary.exists()
        else {}
    )

    solver_rollout_summary_path = round_dir / "summary.json"
    solver_rollout_summary_obj = (
        json_load(solver_rollout_summary_path)
        if solver_rollout_summary_path.exists()
        else {}
    )
    solver_rows_summary_obj = (
        json_load(solver_rows_summary)
        if solver_rows_summary.exists()
        else {}
    )

    return {
        "round": round_idx,
        "route": "rzero",
        "trained": False,
        "challenger_in": challenger_ref,
        "challenger_out": challenger_ref,
        "solver_in": solver_ref,
        "solver_out": solver_ref,
        "challenger_candidates_jsonl": str(candidates_jsonl),
        "challenger_rows_jsonl": str(challenger_rows_jsonl),
        "challenger_tasks_jsonl": str(challenger_tasks_jsonl),
        "solver_traj_jsonl": str(traj_jsonl),
        "solver_rows_jsonl": str(solver_jsonl),
        "challenger_candidate_stats": challenger_candidate_stats,
        "challenger_row_stats": challenger_row_stats,
        "solver_traj_stats": traj_stats,
        "solver_row_stats": row_stats,
        "challenger_rollout_summary": challenger_rollout_summary_obj,
        "challenger_rows_summary": challenger_rows_summary_obj,
        "solver_rollout_summary": solver_rollout_summary_obj,
        "solver_rows_summary": solver_rows_summary_obj,
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.route != "rzero":
        raise ValueError("Initial protocol driver supports only --route rzero")

    if not args.skip_train:
        raise NotImplementedError(
            "Current driver validates challenger->solver data protocol only. Use --skip_train."
        )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    challenger_ref = args.challenger_model or args.challenger_lora or args.base_model or "random_challenger"
    solver_ref = args.solver_model or args.solver_lora or args.base_model or "random_solver"

    rounds: List[Dict[str, Any]] = []

    for r in range(1, int(args.rounds) + 1):
        rec = run_rzero_round(
            args=args,
            round_idx=r,
            challenger_ref=challenger_ref,
            solver_ref=solver_ref,
        )
        rounds.append(rec)

    rounds_path = out_dir / "rounds.json"
    with rounds_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "route": "rzero",
                "env": args.env,
                "protocol": "trainable_challenger_to_monolithic_solver_smoke",
                "trained": False,
                "rounds": rounds,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n[protocol] wrote {rounds_path}")
    print("[protocol] OK: challenger candidate -> challenger rows -> solver rows smoke finished")


if __name__ == "__main__":
    main()
