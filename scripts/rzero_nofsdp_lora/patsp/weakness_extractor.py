# -*- coding: utf-8 -*-
"""PATSP component B: extract the solver's WEAKNESS PROFILE from a batch of rollouts.

Given the trajectories the current solver produced this round, summarise *where and how* it
fails -- programmatically from `StageRecord.failure_type` + per-sub-goal completion. This profile
drives the PATSP challenger (component C), which generates the next round's tasks to TARGET these
weaknesses instead of sampling blindly (R-Zero's challenger is weakness-blind).

Weakness profile:
  failure_dist        : Counter of failure_type over all stage records (none/missing/format/...).
  dominant_failure    : the most frequent NON-"none" failure_type (the solver's main error mode).
  per_pred_failrate   : {predicate_id -> fraction of rollouts where it was never satisfied}.
  bottleneck_preds    : sub-goals with the highest fail-rate that are NOT hopeless (0<rate<1) --
                        the learnable bottlenecks to stress next.
  recovery_rate       : fraction of (broken sub-goal) events that were later re-satisfied.
  mean_subgoal_frac   : avg fraction of sub-goals reached per rollout (competence proxy).

Pure-python, no torch. CLI: weakness_extractor.py --rollouts X.jsonl [--out profile.json].
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Dict, List, Any


def extract_profile(trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    failure_dist: Counter = Counter()
    pred_total: Counter = Counter()       # rollouts that DECLARE each predicate
    pred_unsat: Counter = Counter()       # rollouts where predicate never reached sat
    broken_events = 0
    recovered_events = 0
    subgoal_fracs: List[float] = []

    for tr in trajectories:
        preds = list(tr.get("predicate_library", []) or [])
        pset = set(preds)
        recs = tr.get("stage_records_global", []) or []
        ever_sat: set = set()
        broken: Dict[str, bool] = {}
        for rec in recs:
            ft = str(rec.get("failure_type", "none"))
            failure_dist[ft] += 1
            pid = rec.get("predicate_id")
            before, after = str(rec.get("before", "unsat")), str(rec.get("after", "unsat"))
            if pid is None:
                continue
            if ft != "none" or (before == "sat" and after == "unsat"):
                if not broken.get(pid):
                    broken[pid] = True
                    broken_events += 1
            if after == "sat":
                ever_sat.add(pid)
                if broken.get(pid):
                    recovered_events += 1
                    broken[pid] = False
        for p in preds:
            pred_total[p] += 1
            if p not in ever_sat:
                pred_unsat[p] += 1
        if preds:
            subgoal_fracs.append(len(ever_sat & pset) / len(preds))

    per_pred_failrate = {p: pred_unsat[p] / pred_total[p] for p in pred_total if pred_total[p] > 0}
    # learnable bottlenecks: failing often but not always (0<rate<1), sorted hardest-first
    bottlenecks = sorted([(p, r) for p, r in per_pred_failrate.items() if 0.0 < r < 1.0],
                         key=lambda kv: -kv[1])
    non_none = Counter({k: v for k, v in failure_dist.items() if k != "none"})
    dominant = non_none.most_common(1)[0][0] if non_none else "none"
    return {
        "n_rollouts": len(trajectories),
        "failure_dist": dict(failure_dist),
        "dominant_failure": dominant,
        "per_pred_failrate": per_pred_failrate,
        "bottleneck_preds": [p for p, _ in bottlenecks[:8]],
        "bottleneck_failrate": {p: round(r, 3) for p, r in bottlenecks[:8]},
        "recovery_rate": round(recovered_events / broken_events, 3) if broken_events else 0.0,
        "mean_subgoal_frac": round(sum(subgoal_fracs) / len(subgoal_fracs), 3) if subgoal_fracs else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="PATSP component B: extract solver weakness profile.")
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    trajs = [json.loads(l) for l in open(a.rollouts) if l.strip()]
    prof = extract_profile(trajs)
    if a.out:
        json.dump(prof, open(a.out, "w"), indent=2)
    print(f"[weakness_extractor] {json.dumps(prof)}")


if __name__ == "__main__":
    main()
