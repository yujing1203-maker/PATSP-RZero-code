#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protocol v1 driver for R-Zero vs PATSP experiments.

Current implementation:
  - route=rzero
  - LoRA training mode
  - challenger and solver are both trainable
  - solver is monolithic, not planner/executor

R-Zero round:

  challenger_{i-1}
      -> generate challenger candidates
      -> solver_{i-1} probes candidates
      -> build challenger rows with reward/advantage
      -> train challenger_i

  challenger_i
      -> generate solver-training tasks
      -> solver_{i-1} rollouts
      -> build solver rows
      -> train solver_i

This driver must never implement R-Zero as planner+executor.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
            f"Command prefix: {' '.join(str(x) for x in cmd[:10])}"
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


def copytree_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


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
    p.add_argument("--solver_task_num_prompts", type=int, default=None)
    p.add_argument("--solver_task_samples_per_prompt", type=int, default=None)
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
    p.add_argument("--solver_lora", default=None)

    # Full-parameter checkpoints for R-Zero route.
    p.add_argument("--challenger_model", default=None)
    p.add_argument("--solver_model", default=None)
    p.add_argument(
        "--full_param",
        action="store_true",
        help="Use full-parameter checkpoints instead of LoRA adapters for R-Zero.",
    )

    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)

    # Training controls.
    p.add_argument(
        "--skip_train",
        action="store_true",
        help="Run data protocol only. No LoRA training.",
    )
    p.add_argument("--train_epochs", type=int, default=int(os.environ.get("RZERO_TRAIN_EPOCHS", "1")))
    p.add_argument("--train_lr", default=os.environ.get("RZERO_TRAIN_LR", "2e-6"))
    p.add_argument(
        "--train_backend",
        choices=["single", "fsdp"],
        default=os.environ.get("RZERO_TRAIN_BACKEND", "fsdp"),
        help="Full-param training backend. Use fsdp for two-GPU training.",
    )
    p.add_argument(
        "--fsdp_nproc",
        default=os.environ.get("RZERO_FSDP_NPROC", "2"),
        help="Number of FSDP processes/cards for torch.distributed.run.",
    )
    p.add_argument("--train_max_seq_len", default=os.environ.get("RZERO_TRAIN_MAX_SEQ_LEN", "16384"))
    p.add_argument("--train_max_rows", default=os.environ.get("RZERO_TRAIN_MAX_ROWS", "-1"))
    p.add_argument("--train_min_abs_adv", default=os.environ.get("RZERO_TRAIN_MIN_ABS_ADV", "1e-8"))
    p.add_argument(
        "--objective",
        choices=["pg", "grpo_clip"],
        default="grpo_clip",
    )
    p.add_argument(
        "--credit_mode",
        choices=["uniform", "outcome_only", "stage_hardcoded", "routed"],
        default="uniform",
    )

    # Initial LoRA config.
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

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
        raise RuntimeError(f"R-Zero trajectory contains non-solver roles: {bad_roles}")

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
        raise RuntimeError(f"R-Zero solver rows contain non-solver roles: {bad_roles}")

    return {"n_solver_rows": n_rows}


def append_vllm_common_args(cmd: List[str], args: argparse.Namespace) -> None:
    cmd += [
        "--max_tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--gpu_memory_utilization", str(args.gpu_memory_utilization),
    ]


def init_lora_if_needed(
    *,
    args: argparse.Namespace,
    role: str,
    current_ref: Optional[str],
    out_dir: Path,
    seed: int,
) -> str:
    if current_ref:
        return str(current_ref)

    out = out_dir / f"r0_{role}_lora"
    if (out / "adapter_config.json").exists():
        return str(out)

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "init_lora_adapters.py"),
        "--base_model", args.base_model,
        "--out_dir", str(out),
        "--role", role,
        "--seed", str(seed),
        "--r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", str(args.lora_dropout),
    ]
    run_cmd(cmd, log_path=out_dir / f"r0_init_{role}.log")
    return str(out)



def train_role_full_param(
    *,
    args: argparse.Namespace,
    role: str,
    rollouts_jsonl: Path,
    model_in: str,
    model_out: Path,
    log_path: Path,
) -> str:
    """Train one full-parameter role checkpoint.

    In FSDP mode this launches torch.distributed.run. The training script gathers
    a full state dict and saves a normal HuggingFace checkpoint to model_out.
    """
    if args.train_backend == "fsdp":
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(args.fsdp_nproc),
            str(SCRIPT_DIR / "train_fsdp_grpo_clip_from_rollouts.py"),
            "--base_model", str(model_in),
            "--full_param",
            "--rollouts_jsonl", str(rollouts_jsonl),
            "--lora_out", str(model_out),
            "--objective", args.objective,
            "--epochs", str(args.train_epochs),
            "--credit_mode", args.credit_mode,
            "--max_seq_len", str(args.train_max_seq_len),
            "--max_rows", str(args.train_max_rows),
            "--lr", str(args.train_lr),
            "--min_abs_adv", str(args.train_min_abs_adv),
        ]
    else:
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "train_lora_grpo_clip_from_rollouts.py"),
            "--base_model", str(model_in),
            "--full_param",
            "--rollouts_jsonl", str(rollouts_jsonl),
            "--lora_out", str(model_out),
            "--objective", args.objective,
            "--epochs", str(args.train_epochs),
            "--credit_mode", args.credit_mode,
            "--max_seq_len", str(args.train_max_seq_len),
            "--max_rows", str(args.train_max_rows),
            "--lr", str(args.train_lr),
            "--min_abs_adv", str(args.train_min_abs_adv),
        ]

    try:
        run_cmd(cmd, log_path=log_path)
        return str(model_out)
    except RuntimeError:
        txt = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        flat_markers = [
            "No usable rollout rows",
            "No usable examples",
            "No optimizer update",
            "No usable rows",
        ]
        if any(m in txt for m in flat_markers):
            print(f"[protocol] {role}: flat/no-update full-param round -> carry forward {model_in}")
            return str(model_in)
        raise


def train_role_lora(
    *,
    args: argparse.Namespace,
    role: str,
    rollouts_jsonl: Path,
    lora_in: str,
    lora_out: Path,
    log_path: Path,
) -> str:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "train_lora_grpo_clip_from_rollouts.py"),
        "--base_model", args.base_model,
        "--lora_in", str(lora_in),
        "--rollouts_jsonl", str(rollouts_jsonl),
        "--lora_out", str(lora_out),
        "--objective", args.objective,
        "--epochs", str(args.train_epochs),
        "--credit_mode", args.credit_mode,
        "--max_seq_len", str(args.train_max_seq_len),
        "--max_rows", str(args.train_max_rows),
        "--lr", str(args.train_lr),
        "--min_abs_adv", str(args.train_min_abs_adv),
    ]

    try:
        run_cmd(cmd, log_path=log_path)
        return str(lora_out)
    except RuntimeError:
        txt = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        flat_markers = [
            "No usable rollout rows",
            "No usable examples",
            "No optimizer update",
            "No usable rows",
        ]
        if any(m in txt for m in flat_markers):
            print(f"[protocol] {role}: flat/no-update round -> carry forward {lora_in}")
            copytree_replace(Path(lora_in), lora_out)
            return str(lora_out)
        raise


def build_challenger_candidates(
    *,
    args: argparse.Namespace,
    round_dir: Path,
    round_idx: int,
    challenger_ref: str,
    solver_ref: str,
    tag: str,
    out_candidates_jsonl: Path,
    out_tasks_jsonl: Path,
    samples_per_prompt: int,
    seed: int,
) -> Dict[str, Any]:
    summary_json = round_dir / f"{tag}_challenger_rollouts.summary.json"

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_agentic_challenger_rollouts.py"),
        "--route", "rzero",
        "--env", args.env,
        "--out_candidates_jsonl", str(out_candidates_jsonl),
        "--out_tasks_jsonl", str(out_tasks_jsonl),
        "--summary_json", str(summary_json),
        "--num_prompts", str(args.challenger_num_prompts),
        "--samples_per_prompt", str(samples_per_prompt),
        "--difficulty", str(args.difficulty),
        "--seed", str(seed),
        "--probe_k", str(args.probe_k),
        "--max_steps", str(args.max_steps),
        "--score_min", str(args.score_min),
        "--score_max", str(args.score_max),
        "--probe_score_metric", args.probe_score_metric,
        "--policy", args.policy,
    ]

    if args.dry_run:
        cmd.append("--dry_run")

    append_vllm_common_args(cmd, args)

    if args.policy == "vllm":
        if args.full_param:
            cmd += [
                "--challenger_model", str(challenger_ref),
                "--solver_model", str(solver_ref),
            ]
        else:
            cmd += [
                "--base_model", args.base_model,
                "--challenger_lora", str(challenger_ref),
                "--solver_lora", str(solver_ref),
            ]

    run_cmd(cmd, log_path=round_dir / f"{tag}_challenger_rollouts.log")
    stats = assert_challenger_candidates(out_candidates_jsonl, "rzero")

    if not out_tasks_jsonl.is_file() or count_jsonl(out_tasks_jsonl) == 0:
        raise RuntimeError(
            f"No valid challenger-produced tasks written to {out_tasks_jsonl}. "
            "Use --score_min 0.0 --score_max 1.0 for smoke, or inspect rollout log."
        )

    stats["summary"] = json_load(summary_json) if summary_json.exists() else {}
    return stats


def build_challenger_rows(
    *,
    args: argparse.Namespace,
    round_dir: Path,
    tag: str,
    candidates_jsonl: Path,
    out_rows_jsonl: Path,
) -> Dict[str, Any]:
    summary_json = round_dir / f"{tag}_challenger_rows.summary.json"

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_agentic_challenger_rows.py"),
        "--candidates_jsonl", str(candidates_jsonl),
        "--out_challenger", str(out_rows_jsonl),
        "--summary_json", str(summary_json),
        "--min_score", str(args.score_min),
        "--max_score", str(args.score_max),
        "--require_nonempty",
    ]
    run_cmd(cmd, log_path=round_dir / f"{tag}_challenger_rows.log")

    stats = assert_challenger_rows(out_rows_jsonl)
    stats["summary"] = json_load(summary_json) if summary_json.exists() else {}
    return stats


def build_solver_rollouts_and_rows(
    *,
    args: argparse.Namespace,
    round_dir: Path,
    round_idx: int,
    solver_ref: str,
    tasks_jsonl: Path,
) -> Dict[str, Any]:
    traj_jsonl = round_dir / f"r{round_idx}_traj.jsonl"
    solver_rows_jsonl = round_dir / f"r{round_idx}_solver.jsonl"
    solver_rows_summary = round_dir / f"r{round_idx}_solver_rows.summary.json"

    rollout_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_agentic_solver_rollouts.py"),
        "--env", args.env,
        "--tasks_jsonl", str(tasks_jsonl),
        "--num_tasks", "0",
        "--group_size", str(args.group_size),
        "--max_steps", str(args.max_steps),
        "--out_jsonl", str(traj_jsonl),
        "--policy", args.policy,
        "--split", args.split,
        "--seed", str(args.seed + round_idx * 1000),
    ]

    if args.dry_run:
        rollout_cmd.append("--dry_run")

    append_vllm_common_args(rollout_cmd, args)

    if args.policy == "vllm":
        if args.full_param:
            rollout_cmd += [
                "--solver_model", str(solver_ref),
            ]
        else:
            rollout_cmd += [
                "--base_model", args.base_model,
                "--solver_lora", str(solver_ref),
            ]

    run_cmd(rollout_cmd, log_path=round_dir / f"r{round_idx}_solver_rollouts.log")
    traj_stats = assert_solver_only_trajectories(traj_jsonl)

    rows_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_agentic_solver_rows.py"),
        "--trajectories_jsonl", str(traj_jsonl),
        "--out_solver", str(solver_rows_jsonl),
        "--summary_json", str(solver_rows_summary),
        "--require_nonempty",
    ]
    run_cmd(rows_cmd, log_path=round_dir / f"r{round_idx}_solver_rows.log")
    row_stats = assert_solver_rows(solver_rows_jsonl)

    return {
        "solver_traj_jsonl": str(traj_jsonl),
        "solver_rows_jsonl": str(solver_rows_jsonl),
        "solver_traj_stats": traj_stats,
        "solver_row_stats": row_stats,
        "solver_rows_summary": json_load(solver_rows_summary) if solver_rows_summary.exists() else {},
    }


def run_rzero_round(
    *,
    args: argparse.Namespace,
    round_idx: int,
    challenger_ref: str,
    solver_ref: str,
) -> Dict[str, Any]:
    out_dir = Path(args.out_dir).resolve()
    round_dir = out_dir / f"r{round_idx}"
    round_dir.mkdir(parents=True, exist_ok=True)

    # A. challenger_{i-1} candidates -> challenger rows.
    challenger_candidates_jsonl = round_dir / f"r{round_idx}_challenger_candidates.jsonl"
    challenger_rows_jsonl = round_dir / f"r{round_idx}_challenger.jsonl"
    pretrain_tasks_jsonl = round_dir / f"r{round_idx}_challenger_pretrain_tasks.jsonl"

    cand_stats = build_challenger_candidates(
        args=args,
        round_dir=round_dir,
        round_idx=round_idx,
        challenger_ref=challenger_ref,
        solver_ref=solver_ref,
        tag=f"r{round_idx}_challenger",
        out_candidates_jsonl=challenger_candidates_jsonl,
        out_tasks_jsonl=pretrain_tasks_jsonl,
        samples_per_prompt=args.challenger_samples_per_prompt,
        seed=args.seed + round_idx * 100,
    )

    challenger_row_stats = build_challenger_rows(
        args=args,
        round_dir=round_dir,
        tag=f"r{round_idx}",
        candidates_jsonl=challenger_candidates_jsonl,
        out_rows_jsonl=challenger_rows_jsonl,
    )

    # B. train challenger_i.
    challenger_out = round_dir / (
        f"r{round_idx}_challenger_model" if args.full_param else f"r{round_idx}_challenger_lora"
    )
    if args.skip_train:
        challenger_next = challenger_ref
        challenger_trained = False
    else:
        if args.full_param:
            challenger_next = train_role_full_param(
                args=args,
                role="challenger",
                rollouts_jsonl=challenger_rows_jsonl,
                model_in=challenger_ref,
                model_out=challenger_out,
                log_path=round_dir / f"r{round_idx}_train_challenger.log",
            )
        else:
            challenger_next = train_role_lora(
                args=args,
                role="challenger",
                rollouts_jsonl=challenger_rows_jsonl,
                lora_in=challenger_ref,
                lora_out=challenger_out,
                log_path=round_dir / f"r{round_idx}_train_challenger.log",
            )
        challenger_trained = True

    # C. updated challenger_i produces solver-training tasks.
    solver_task_num_prompts = (
        args.solver_task_num_prompts
        if args.solver_task_num_prompts is not None
        else args.challenger_num_prompts
    )
    solver_task_samples_per_prompt = (
        args.solver_task_samples_per_prompt
        if args.solver_task_samples_per_prompt is not None
        else args.challenger_samples_per_prompt
    )

    # build_challenger_candidates uses args.challenger_num_prompts internally;
    # override only for this call by shallow-copying the namespace.
    solver_task_args = argparse.Namespace(**vars(args))
    solver_task_args.challenger_num_prompts = solver_task_num_prompts

    solver_task_candidates_jsonl = round_dir / f"r{round_idx}_solver_task_candidates.jsonl"
    solver_tasks_jsonl = round_dir / f"r{round_idx}_tasks.jsonl"

    solver_task_stats = build_challenger_candidates(
        args=solver_task_args,
        round_dir=round_dir,
        round_idx=round_idx,
        challenger_ref=challenger_next,
        solver_ref=solver_ref,
        tag=f"r{round_idx}_solver_tasks",
        out_candidates_jsonl=solver_task_candidates_jsonl,
        out_tasks_jsonl=solver_tasks_jsonl,
        samples_per_prompt=solver_task_samples_per_prompt,
        seed=args.seed + round_idx * 200,
    )

    # D. solver_{i-1} rollout on challenger_i tasks -> solver rows.
    solver_data = build_solver_rollouts_and_rows(
        args=args,
        round_dir=round_dir,
        round_idx=round_idx,
        solver_ref=solver_ref,
        tasks_jsonl=solver_tasks_jsonl,
    )

    solver_rows_jsonl = Path(solver_data["solver_rows_jsonl"])

    # E. train solver_i.
    solver_out = round_dir / (
        f"r{round_idx}_solver_model" if args.full_param else f"r{round_idx}_solver_lora"
    )
    if args.skip_train:
        solver_next = solver_ref
        solver_trained = False
    else:
        if args.full_param:
            solver_next = train_role_full_param(
                args=args,
                role="solver",
                rollouts_jsonl=solver_rows_jsonl,
                model_in=solver_ref,
                model_out=solver_out,
                log_path=round_dir / f"r{round_idx}_train_solver.log",
            )
        else:
            solver_next = train_role_lora(
                args=args,
                role="solver",
                rollouts_jsonl=solver_rows_jsonl,
                lora_in=solver_ref,
                lora_out=solver_out,
                log_path=round_dir / f"r{round_idx}_train_solver.log",
            )
        solver_trained = True

    return {
        "round": round_idx,
        "route": "rzero",
        "trained": bool(challenger_trained or solver_trained),
        "challenger_trained": challenger_trained,
        "solver_trained": solver_trained,
        "challenger_in": challenger_ref,
        "challenger_out": challenger_next,
        "solver_in": solver_ref,
        "solver_out": solver_next,
        "challenger_candidates_jsonl": str(challenger_candidates_jsonl),
        "challenger_rows_jsonl": str(challenger_rows_jsonl),
        "challenger_pretrain_tasks_jsonl": str(pretrain_tasks_jsonl),
        "solver_task_candidates_jsonl": str(solver_task_candidates_jsonl),
        "challenger_tasks_jsonl": str(solver_tasks_jsonl),
        "challenger_candidate_stats": cand_stats,
        "challenger_row_stats": challenger_row_stats,
        "solver_task_stats": solver_task_stats,
        **solver_data,
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.route != "rzero":
        raise ValueError("Current protocol driver supports only --route rzero")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        if args.policy != "vllm":
            raise ValueError("Real training requires --policy vllm")
        if not args.base_model and not args.full_param:
            raise ValueError("LoRA training requires --base_model or BASE_MODEL env var")
        if args.full_param and not (args.base_model or args.challenger_model or args.solver_model):
            raise ValueError("Full-param training requires --base_model or role model paths")

    if args.full_param:
        challenger_ref = args.challenger_model or args.base_model or "random_challenger_model"
        solver_ref = args.solver_model or args.base_model or "random_solver_model"
    elif args.skip_train:
        challenger_ref = args.challenger_lora or args.base_model or "random_challenger"
        solver_ref = args.solver_lora or args.base_model or "random_solver"
    else:
        challenger_ref = init_lora_if_needed(
            args=args,
            role="challenger",
            current_ref=args.challenger_lora,
            out_dir=out_dir,
            seed=args.seed + 11,
        )
        solver_ref = init_lora_if_needed(
            args=args,
            role="solver",
            current_ref=args.solver_lora,
            out_dir=out_dir,
            seed=args.seed + 22,
        )

    rounds: List[Dict[str, Any]] = []

    for r in range(1, int(args.rounds) + 1):
        rec = run_rzero_round(
            args=args,
            round_idx=r,
            challenger_ref=challenger_ref,
            solver_ref=solver_ref,
        )
        rounds.append(rec)

        challenger_ref = rec["challenger_out"]
        solver_ref = rec["solver_out"]

    rounds_path = out_dir / "rounds.json"
    with rounds_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "route": "rzero",
                "env": args.env,
                "protocol": (
                    "trainable_challenger_monolithic_solver_full_param"
                    if args.full_param
                    else "trainable_challenger_monolithic_solver_lora"
                ),
                "trained": not args.skip_train,
                "rounds": rounds,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n[protocol] wrote {rounds_path}")
    print("[protocol] OK: R-Zero challenger-solver protocol finished")


if __name__ == "__main__":
    main()