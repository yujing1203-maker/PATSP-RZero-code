#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build monolithic R-Zero solver training rows from solver-only agentic trajectories.

Input:
  trajectories JSONL emitted by build_agentic_solver_rollouts.py

Required trajectory turn contract:
  every model turn used for training must have:
      role == "solver"

This script intentionally rejects planner/executor turns by default. That prevents
the R-Zero baseline from accidentally becoming a planner/executor model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# Robust local imports.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from split_agentic_rollouts_by_role import (  # noqa: E402
    compute_group_advantages_over_terminal,
    build_role_row,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build R-Zero monolithic solver training rows from role=solver trajectories."
    )
    p.add_argument(
        "--trajectories_jsonl",
        required=True,
        help="Input solver-only trajectories JSONL.",
    )
    p.add_argument(
        "--out_solver",
        required=True,
        help="Output solver training JSONL.",
    )
    p.add_argument(
        "--summary_json",
        default=None,
        help="Optional summary JSON path. Default: out_solver with .summary.json suffix.",
    )
    p.add_argument(
        "--require_nonempty",
        action="store_true",
        help="Fail if no solver rows are emitted.",
    )
    p.add_argument(
        "--allow_non_solver",
        action="store_true",
        help="Skip non-solver turns instead of failing. Off by default for protocol safety.",
    )
    return p.parse_args(argv)


def load_trajectories(path: Path) -> List[Dict[str, Any]]:
    trajectories: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path}:{lineno}: {e}") from e

            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object in {path}:{lineno}")

            trajectories.append(obj)

    return trajectories


def build_solver_row(
    traj: Dict[str, Any],
    turn: Dict[str, Any],
    outcome_advantage: float,
) -> Dict[str, Any]:
    role = str(turn.get("role", ""))
    if role != "solver":
        raise ValueError(f"build_solver_row expected role='solver', got role={role!r}")

    row = build_role_row(
        traj=traj,
        turn=turn,
        outcome_advantage=outcome_advantage,
        planner_advantage=None,
    )

    # Make the monolithic-solver semantics explicit and stable.
    row["source"] = "agentic_solver"
    row["role"] = "solver"
    row["solver_role"] = "solver"
    row["stage_advantages"] = None

    # R-Zero solver uses outcome-level GRPO by default.
    row["advantage"] = float(outcome_advantage)
    row["outcome_advantage"] = float(outcome_advantage)

    row["note"] = (
        f"agentic monolithic_solver role=solver "
        f"turn_index={turn.get('turn_index')} boundary={turn.get('boundary')}"
    )

    return row


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    in_path = Path(args.trajectories_jsonl)
    if not in_path.is_file():
        raise FileNotFoundError(f"--trajectories_jsonl not found: {in_path}")

    out_path = Path(args.out_solver)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary_path = (
        Path(args.summary_json)
        if args.summary_json
        else out_path.with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    trajectories = load_trajectories(in_path)
    adv_by_key = compute_group_advantages_over_terminal(trajectories)

    n_rows = 0
    n_empty_completion = 0
    n_non_solver_turns = 0
    rows_by_role: Dict[str, int] = defaultdict(int)
    bad_roles: Dict[str, int] = defaultdict(int)

    with out_path.open("w", encoding="utf-8") as f:
        for traj in trajectories:
            key = (
                str(traj.get("task_id", "")),
                int(traj.get("sample_index", 0)),
            )
            outcome_advantage = adv_by_key.get(key, 0.0)

            for turn in traj.get("turns", []) or []:
                role = str(turn.get("role", ""))

                if role != "solver":
                    n_non_solver_turns += 1
                    bad_roles[role] += 1

                    if not args.allow_non_solver:
                        raise RuntimeError(
                            "Non-solver turn found in R-Zero solver trajectory. "
                            f"role={role!r}. This script only accepts role='solver'. "
                            "Use build_agentic_solver_rollouts.py, not planner/executor rollouts."
                        )
                    continue

                completion = str(turn.get("completion", ""))
                if not completion.strip():
                    n_empty_completion += 1

                row = build_solver_row(traj, turn, outcome_advantage)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

                n_rows += 1
                rows_by_role[role] += 1

    if args.require_nonempty and n_rows == 0:
        raise RuntimeError(f"No solver rows emitted from {in_path}")

    summary = {
        "trajectories_jsonl": str(in_path),
        "out_solver": str(out_path),
        "n_trajectories": len(trajectories),
        "n_solver_rows": n_rows,
        "rows_by_role": dict(rows_by_role),
        "n_non_solver_turns": n_non_solver_turns,
        "bad_roles": dict(bad_roles),
        "n_empty_completion": n_empty_completion,
        "strict_role_contract": "role must be solver unless --allow_non_solver is set",
        "route": "rzero_monolithic_solver",
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[build_agentic_solver_rows]", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
