#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build trainable challenger GRPO rows from challenger task candidates.

This file is for the TRUE trainable Challenger in R-Zero/PATSP.

Important distinction:
  - generate_agentic_challenger_tasks.py is a task sampler/gater.
  - this file consumes MODEL-GENERATED challenger candidates and turns their
    probe results into policy-gradient training rows.

A challenger candidate is expected to contain:
  {
    "uid": "...",                       # group key, usually prompt/task-family id
    "sample_index": 0,                  # rollout sample index within group
    "prompt": "...",                    # challenger prompt
    "completion": "<task>{...}</task>", # challenger model output
    "format_ok": true,
    "well_posed": true,
    "task": {...},                      # parsed Task.to_dict()
    "probe_route": "rzero" | "patsp",
    "probe_score": 0.5,                 # mean terminal reward or solve rate
    "probe_rewards": [0.0, 1.0]
  }

Reward semantics:
  - valid format and well-posed task are required.
  - edge-of-ability reward follows R-Zero:
        min(score, 1 - score)
    so too-easy and too-hard tasks get low reward.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build challenger training rows from model-generated task candidates."
    )
    p.add_argument("--candidates_jsonl", required=True)
    p.add_argument("--out_challenger", required=True)
    p.add_argument("--summary_json", default=None)
    p.add_argument("--score_key", default="probe_score")
    p.add_argument("--min_score", type=float, default=0.0)
    p.add_argument("--max_score", type=float, default=1.0)
    p.add_argument("--invalid_reward", type=float, default=-1.0)
    p.add_argument("--format_weight", type=float, default=0.0)
    p.add_argument(
        "--require_nonempty",
        action="store_true",
        help="Fail if no challenger rows are emitted.",
    )
    return p.parse_args(argv)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{lineno}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{lineno}")
            rows.append(obj)
    return rows


def clamp01(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def challenger_reward(row: Dict[str, Any], args: argparse.Namespace) -> Tuple[float, Dict[str, Any]]:
    """Compute trainable challenger reward from candidate probe result."""
    format_ok = bool(row.get("format_ok", False))
    well_posed = bool(row.get("well_posed", False))
    has_task = isinstance(row.get("task"), dict)

    score_raw = row.get(args.score_key, row.get("probe_score", 0.0))
    try:
        score = clamp01(float(score_raw))
    except Exception:
        score = 0.0

    in_band = args.min_score <= score <= args.max_score

    if not format_ok or not well_posed or not has_task:
        reward = float(args.invalid_reward)
        reason = "invalid"
    elif not in_band:
        # Candidate is valid but outside the chosen learnability band.
        # Keep it trainable with low edge reward rather than crashing.
        reward = min(score, 1.0 - score)
        reason = "out_of_band"
    else:
        reward = min(score, 1.0 - score)
        reason = "edge_of_ability"

    if args.format_weight and format_ok:
        reward += float(args.format_weight)

    meta = {
        "format_ok": format_ok,
        "well_posed": well_posed,
        "has_task": has_task,
        "probe_score": score,
        "in_band": in_band,
        "reward_reason": reason,
    }
    return float(reward), meta


def group_key(row: Dict[str, Any]) -> str:
    return str(
        row.get("uid")
        or row.get("prompt_uid")
        or row.get("task_family")
        or row.get("env")
        or "challenger_group"
    )


def compute_advantages(rows: List[Dict[str, Any]]) -> Dict[int, float]:
    by_group: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_group[group_key(row)].append((i, float(row["reward"])))

    adv: Dict[int, float] = {}
    for _, vals in by_group.items():
        mean = sum(v for _, v in vals) / max(1, len(vals))
        for i, v in vals:
            adv[i] = float(v - mean)
    return adv


def build_train_row(src: Dict[str, Any], reward: float, advantage: float, meta: Dict[str, Any]) -> Dict[str, Any]:
    prompt = str(src.get("prompt") or "")
    completion = str(src.get("completion") or src.get("raw_output") or "")

    return {
        "uid": group_key(src),
        "sample_index": int(src.get("sample_index", 0)),
        "role": "challenger",
        "source": "agentic_challenger",
        "prompt": prompt,
        "completion": completion,
        "completion_for_train": completion,
        "reward": float(reward),
        "advantage": float(advantage),
        "outcome_advantage": float(advantage),
        "format_ok": bool(meta["format_ok"]),
        "well_posed": bool(meta["well_posed"]),
        "probe_route": str(src.get("probe_route", "")),
        "probe_score": float(meta["probe_score"]),
        "probe_rewards": src.get("probe_rewards", []),
        "task": src.get("task"),
        "reward_reason": meta["reward_reason"],
        "note": "trainable challenger row; reward=min(probe_score,1-probe_score) for valid tasks",
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    in_path = Path(args.candidates_jsonl)
    out_path = Path(args.out_challenger)
    summary_path = Path(args.summary_json) if args.summary_json else out_path.with_suffix(".summary.json")

    if not in_path.is_file():
        raise FileNotFoundError(in_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = load_jsonl(in_path)

    scored: List[Dict[str, Any]] = []
    n_invalid = 0
    n_out_of_band = 0
    n_valid = 0

    for c in candidates:
        reward, meta = challenger_reward(c, args)
        c2 = dict(c)
        c2["reward"] = reward
        c2["_reward_meta"] = meta
        scored.append(c2)

        if meta["reward_reason"] == "invalid":
            n_invalid += 1
        elif meta["reward_reason"] == "out_of_band":
            n_out_of_band += 1
        else:
            n_valid += 1

    adv = compute_advantages(scored)

    n_rows = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(scored):
            row = build_train_row(
                c,
                reward=float(c["reward"]),
                advantage=float(adv.get(i, 0.0)),
                meta=c["_reward_meta"],
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_rows += 1

    if args.require_nonempty and n_rows == 0:
        raise RuntimeError("No challenger rows emitted.")

    summary = {
        "candidates_jsonl": str(in_path),
        "out_challenger": str(out_path),
        "n_candidates": len(candidates),
        "n_rows": n_rows,
        "n_valid_edge": n_valid,
        "n_invalid": n_invalid,
        "n_out_of_band": n_out_of_band,
        "reward": "valid ? min(probe_score, 1-probe_score) : invalid_reward",
        "role": "challenger",
        "source": "agentic_challenger",
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[build_agentic_challenger_rows]", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
