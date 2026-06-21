# -*- coding: utf-8 -*-
"""PATSP component A' : OUTCOME-GROUNDED dense credit (the fix for hand-crafted shaping).

Hand-crafted dense reward (sub-goal / recovery counting) is a gameable proxy: GRPO farms sub-goal
touches instead of completing the task -> collapse (Goodhart). This module instead LEARNS the dense
signal FROM THE ROLLOUT OUTCOMES of the round itself, so it can only credit progress that actually
predicts success.

Method (no extra network -- a tabular Monte-Carlo value over progress buckets):
  1. For each rollout, progress = fraction of the task's sub-goals satisfied AT THE END
     (from StageRecords), and R = terminal_reward (the TRUE outcome).
  2. Estimate V(progress) = mean R over all rollouts in that progress bucket (pooled across the
     batch). V is the outcome-grounded value of a progress level: "if you reach this much progress,
     how often do you actually succeed?"
  3. shaped = R + lambda * (V(progress) - V(0)).
     - If progress PREDICTS success (V increasing), partial-progress rollouts get aligned dense
       credit -> non-zero group-advantage even when all fail the final goal.
     - If progress does NOT predict success (farming useless sub-goals -> V flat/~0), the bonus
       vanishes -> NOT gameable. Worst case shaped ~= R (>= R-Zero, never a hand-crafted-shaping collapse).

This is a learned, telescoping-free, per-rollout potential whose potential = the empirical success
value of the reached state -- unlike a hand-crafted sub-goal COUNT potential, it cannot be farmed.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Any


def _progress(row: Dict[str, Any]) -> float:
    preds = set(row.get("predicate_library", []) or [])
    n = len(preds)
    if not n:
        return 0.0
    final: Dict[str, str] = {}
    for rec in row.get("stage_records_global", []) or []:
        pid = rec.get("predicate_id")
        if pid is not None:
            final[pid] = str(rec.get("after", "unsat"))
    k = sum(1 for p in preds if final.get(p) == "sat")
    return k / n


def rewrite_rollouts(in_path: str, out_path: str, lam: float = 0.5, n_buckets: int = 6) -> Dict[str, Any]:
    rows = [json.loads(l) for l in open(in_path) if l.strip()]
    grid = max(1, n_buckets - 1)

    def bkt(p: float) -> float:
        return round(p * grid) / grid          # snap progress to a coarse grid

    # 1-2: outcome-grounded value table V(progress bucket) = mean terminal_reward
    by_bucket: Dict[float, List[float]] = defaultdict(list)
    for r in rows:
        by_bucket[bkt(_progress(r))].append(float(r.get("terminal_reward", 0.0) or 0.0))
    V = {b: (sum(v) / len(v)) for b, v in by_bucket.items()}
    V0 = V.get(bkt(0.0), 0.0)

    # 3: shaped reward (outcome-grounded potential)
    deltas = []
    with open(out_path, "w") as fout:
        for r in rows:
            R = float(r.get("terminal_reward", 0.0) or 0.0)
            p = _progress(r)
            sh = R + lam * (V.get(bkt(p), 0.0) - V0)
            r["terminal_reward_raw"] = R
            r["terminal_reward"] = sh
            deltas.append(sh - R)
            fout.write(json.dumps(r) + "\n")
    return {"rows": len(rows), "lambda": lam,
            "V_by_progress": {round(b, 3): round(v, 4) for b, v in sorted(V.items())},
            "mean_shaped_minus_raw": round(sum(deltas) / len(deltas), 4) if deltas else 0.0}


def main():
    ap = argparse.ArgumentParser(description="PATSP A': outcome-grounded dense credit rewrite.")
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--lam", type=float, default=float(os.environ.get("PATSP_OUTCOME_LAMBDA", "0.5")))
    ap.add_argument("--n_buckets", type=int, default=6)
    a = ap.parse_args()
    print(f"[outcome_credit] {json.dumps(rewrite_rollouts(a.in_jsonl, a.out_jsonl, a.lam, a.n_buckets))}")


if __name__ == "__main__":
    main()
