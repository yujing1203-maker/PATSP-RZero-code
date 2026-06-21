# -*- coding: utf-8 -*-
"""PATSP component C: weakness-targeted adversarial challenger.

R-Zero's challenger samples tasks blindly within a learnability band. PATSP's challenger reads the
solver's WEAKNESS PROFILE (component B) and steers the NEXT round's task distribution toward the
solver's actual failure mode -- "progressive adversarial": as the solver fixes one weakness the
profile shifts and the challenger chases the next.

Decision (per round, from the profile of the round's own rollouts):
  competence = mean_subgoal_frac (how far the solver gets on average)
  * If the solver is COMPETENT (competence high) but its dominant error is an ORDERING/PREREQUISITE
    mistake (missing / constraint_violation) -> it has outgrown the current horizon; push DEEPER
    (raise difficulty -> longer dependency chains that stress ordering). This is the adversarial move.
  * If the solver is STRUGGLING (competence low) -> hold/ease (stay in the learnable band).
  * Otherwise hold.
Also surfaces the learnable bottleneck sub-goals (for envs whose sampler can target them).

This is weakness-CONDITIONED curriculum (vs ADP-LP's blind too-easy-fraction): the difficulty move
is gated on the *type* of failure, and only fires when the failure is a learnable ordering error
rather than blanket incompetence. Pure-python, no torch.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple

# failure types that mean "the solver acts but sequences wrong" -> push horizon
_ORDERING_FAILURES = {"missing", "constraint_violation"}


def next_difficulty(cur_d: float, profile: Dict[str, Any], step: float = 0.15,
                    comp_hi: float = 0.55, comp_lo: float = 0.25,
                    d_max: float = 0.95, d_min: float = 0.1) -> Tuple[float, str]:
    """Return (next_difficulty, reason) from the weakness profile."""
    comp = float(profile.get("mean_subgoal_frac", 0.0))
    dom = str(profile.get("dominant_failure", "none"))
    if comp >= comp_hi and dom in _ORDERING_FAILURES:
        return (min(d_max, cur_d + step),
                f"competent(comp={comp:.2f}) + ordering-failure({dom}) -> push deeper")
    if comp < comp_lo:
        return (max(d_min, cur_d - step),
                f"struggling(comp={comp:.2f}) -> ease back into learnable band")
    return (cur_d, f"hold(comp={comp:.2f}, dom={dom})")


def target_hint(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Generation bias for envs whose sampler can consume it (e.g. include bottleneck sub-goals)."""
    return {"bottleneck_preds": profile.get("bottleneck_preds", []),
            "dominant_failure": profile.get("dominant_failure", "none")}
