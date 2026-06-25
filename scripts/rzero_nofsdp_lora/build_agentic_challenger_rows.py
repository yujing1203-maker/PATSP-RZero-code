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

R-Zero-style challenger reward:
  - Format gate:
      malformed challenger outputs receive final reward 0.
  - Uncertainty reward:
      uncertainty = 1 - 2 * abs(score - 0.5)
      maximized when the current Solver / Planner-Executor is uncertain.
  - Repetition penalty:
      repeated/similar tasks in the same challenger group are penalized.
  - Composite reward:
      reward = max(0, uncertainty_reward - repetition_penalty)
  - Advantages:
      group-relative z-score advantages are used for GRPO-style updates.

Environment adaptation:
  Original R-Zero checks <question> format. Here the challenger emits
  <task>{...}</task>, so format_ok and well_posed are produced upstream by
  build_agentic_challenger_rollouts.py.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict, Counter
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

    # Kept for compatibility with older scripts, but default is now 0.0 to
    # match the R-Zero format-fail reward behavior more closely.
    p.add_argument("--invalid_reward", type=float, default=0.0)
    p.add_argument("--format_fail_reward", type=float, default=0.0)

    # R-Zero repetition penalty uses a BLEU-style similarity threshold.
    # We implement a lightweight n-gram BLEU-like similarity to avoid adding
    # a new runtime dependency.
    p.add_argument("--repetition_lambda", type=float, default=1.0)
    p.add_argument("--repetition_bleu_threshold", type=float, default=0.5)
    p.add_argument("--repetition_max_ngram", type=int, default=4)

    # Deprecated; retained so existing command lines do not break.
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


def tokenize_text(s: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", s.lower())
    return toks or list(s.strip().lower())


def ngram_counts(tokens: List[str], n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_like_similarity(a: str, b: str, max_ngram: int = 4) -> float:
    """Small dependency-free BLEU-like similarity in [0, 1].

    This is used only for repetition penalty clustering. It intentionally favors
    precision-like n-gram overlap and includes a brevity penalty.
    """
    ta = tokenize_text(a)
    tb = tokenize_text(b)
    if not ta or not tb:
        return 0.0

    max_ngram = max(1, int(max_ngram))
    precisions: List[float] = []
    for n in range(1, max_ngram + 1):
        ca = ngram_counts(ta, n)
        cb = ngram_counts(tb, n)
        total = sum(ca.values())
        if total <= 0:
            continue
        overlap = sum(min(v, cb.get(k, 0)) for k, v in ca.items())
        # Add tiny smoothing so 4-gram mismatch does not zero everything.
        precisions.append((overlap + 1e-9) / (total + 1e-9))

    if not precisions:
        return 0.0

    log_p = sum(math.log(max(p, 1e-12)) for p in precisions) / len(precisions)
    bp = 1.0 if len(ta) >= len(tb) else math.exp(1.0 - len(tb) / max(1, len(ta)))
    return clamp01(bp * math.exp(log_p))


def group_key(row: Dict[str, Any]) -> str:
    return str(
        row.get("uid")
        or row.get("prompt_uid")
        or row.get("task_family")
        or row.get("env")
        or "challenger_group"
    )


def task_text(row: Dict[str, Any]) -> str:
    task = row.get("task")
    if isinstance(task, dict):
        return json.dumps(task, sort_keys=True, ensure_ascii=False)
    return str(row.get("completion") or row.get("raw_output") or "")


def base_reward_meta(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    format_ok = bool(row.get("format_ok", False))
    well_posed = bool(row.get("well_posed", False))
    has_task = isinstance(row.get("task"), dict)

    score_raw = row.get(args.score_key, row.get("probe_score", 0.0))
    try:
        score = clamp01(float(score_raw))
    except Exception:
        score = 0.0

    in_band = args.min_score <= score <= args.max_score
    uncertainty_reward = clamp01(1.0 - 2.0 * abs(score - 0.5))

    return {
        "format_ok": format_ok,
        "well_posed": well_posed,
        "has_task": has_task,
        "probe_score": score,
        "in_band": in_band,
        "uncertainty_reward": uncertainty_reward,
        "repetition_penalty": 0.0,
        "reward_reason": "",
    }


def compute_repetition_penalties(
    candidates: List[Dict[str, Any]],
    metas: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[int, float]:
    """Cluster similar valid candidate tasks per challenger group.

    R-Zero describes a repetition penalty proportional to the relative cluster
    size. Here cluster membership is computed by connected components over
    BLEU-like similarity >= repetition_bleu_threshold.
    """
    penalties: Dict[int, float] = {i: 0.0 for i in range(len(candidates))}
    by_group: Dict[str, List[int]] = defaultdict(list)

    for i, row in enumerate(candidates):
        by_group[group_key(row)].append(i)

    for _, idxs in by_group.items():
        batch_size = max(1, len(idxs))
        valid_idxs = [
            i
            for i in idxs
            if metas[i]["format_ok"] and metas[i]["well_posed"] and metas[i]["has_task"]
        ]

        if not valid_idxs:
            continue

        parent = {i: i for i in valid_idxs}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        texts = {i: task_text(candidates[i]) for i in valid_idxs}
        for pos, i in enumerate(valid_idxs):
            for j in valid_idxs[pos + 1 :]:
                sim = bleu_like_similarity(
                    texts[i],
                    texts[j],
                    max_ngram=args.repetition_max_ngram,
                )
                if sim >= args.repetition_bleu_threshold:
                    union(i, j)

        clusters: Dict[int, List[int]] = defaultdict(list)
        for i in valid_idxs:
            clusters[find(i)].append(i)

        for members in clusters.values():
            penalty = float(args.repetition_lambda) * (len(members) / batch_size)
            for i in members:
                penalties[i] = penalty

    return penalties


def finalize_reward(meta: Dict[str, Any], args: argparse.Namespace) -> Tuple[float, str]:
    if not meta["format_ok"]:
        return float(args.format_fail_reward), "format_fail"

    if not meta["well_posed"] or not meta["has_task"]:
        return float(args.invalid_reward), "invalid_task"

    reward = max(
        0.0,
        float(meta["uncertainty_reward"]) - float(meta["repetition_penalty"]),
    )

    if args.format_weight and meta["format_ok"]:
        reward += float(args.format_weight)

    if not meta["in_band"]:
        reason = "valid_out_of_band_composite"
    else:
        reason = "valid_composite"

    return float(reward), reason


def compute_advantages(rows: List[Dict[str, Any]]) -> Dict[int, float]:
    by_group: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_group[group_key(row)].append((i, float(row["reward"])))

    adv: Dict[int, float] = {}
    for _, vals in by_group.items():
        mean = sum(v for _, v in vals) / max(1, len(vals))
        var = sum((v - mean) ** 2 for _, v in vals) / max(1, len(vals))
        std = math.sqrt(var)
        for i, v in vals:
            adv[i] = float((v - mean) / (std + 1e-6)) if std > 0 else 0.0
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
        "has_task": bool(meta["has_task"]),
        "probe_route": str(src.get("probe_route", "")),
        "probe_score": float(meta["probe_score"]),
        "probe_rewards": src.get("probe_rewards", []),
        "task": src.get("task"),
        "in_band": bool(meta["in_band"]),
        "uncertainty_reward": float(meta["uncertainty_reward"]),
        "repetition_penalty": float(meta["repetition_penalty"]),
        "reward_reason": meta["reward_reason"],
        "note": (
            "trainable challenger row; R-Zero-style composite reward: "
            "format gate + uncertainty_reward - repetition_penalty"
        ),
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
    metas = [base_reward_meta(c, args) for c in candidates]
    penalties = compute_repetition_penalties(candidates, metas, args)

    scored: List[Dict[str, Any]] = []
    reason_counts: Counter = Counter()
    n_invalid = 0
    n_out_of_band = 0
    n_valid = 0

    for i, c in enumerate(candidates):
        meta = dict(metas[i])
        meta["repetition_penalty"] = float(penalties.get(i, 0.0))
        reward, reason = finalize_reward(meta, args)
        meta["reward_reason"] = reason

        c2 = dict(c)
        c2["reward"] = reward
        c2["_reward_meta"] = meta
        scored.append(c2)

        reason_counts[reason] += 1
        if reason in {"format_fail", "invalid_task"}:
            n_invalid += 1
        elif reason == "valid_out_of_band_composite":
            n_out_of_band += 1
            n_valid += 1
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

    rewards = [float(x["reward"]) for x in scored]
    advantages = [float(adv.get(i, 0.0)) for i in range(len(scored))]
    repetition_penalties = [float(x["_reward_meta"]["repetition_penalty"]) for x in scored]
    uncertainty_rewards = [float(x["_reward_meta"]["uncertainty_reward"]) for x in scored]

    summary = {
        "candidates_jsonl": str(in_path),
        "out_challenger": str(out_path),
        "n_candidates": len(candidates),
        "n_rows": n_rows,
        "n_valid_composite": n_valid,
        "n_invalid": n_invalid,
        "n_out_of_band": n_out_of_band,
        "reward": "format_ok && well_posed ? max(0, uncertainty_reward - repetition_penalty) : 0",
        "uncertainty_reward": "1 - 2 * abs(probe_score - 0.5)",
        "repetition_penalty": "repetition_lambda * cluster_size / batch_size",
        "repetition_lambda": args.repetition_lambda,
        "repetition_bleu_threshold": args.repetition_bleu_threshold,
        "reason_counts": dict(reason_counts),
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "advantage_min": min(advantages) if advantages else None,
        "advantage_max": max(advantages) if advantages else None,
        "uncertainty_reward_min": min(uncertainty_rewards) if uncertainty_rewards else None,
        "uncertainty_reward_max": max(uncertainty_rewards) if uncertainty_rewards else None,
        "repetition_penalty_min": min(repetition_penalties) if repetition_penalties else None,
        "repetition_penalty_max": max(repetition_penalties) if repetition_penalties else None,
        "role": "challenger",
        "source": "agentic_challenger",
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[build_agentic_challenger_rows]", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
