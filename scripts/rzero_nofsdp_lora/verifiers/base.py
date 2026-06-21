# -*- coding: utf-8 -*-
"""
Verifier contract base types.

Defines:
  - StageRecord: per-stage verifier record. char_span indexes into the
    rollout `completion` string and is consumed by the trainer
    (train_lora_grpo_clip_from_rollouts.build_token_advantages) to map stage
    spans onto completion tokens.
  - StageVerifier: abstract base class. A concrete verifier exposes a
    predicate library L_env for a problem and verifies a completion against it.
  - sigma_key: stable string for the frozenset of satisfied predicate_ids
    up to (and including) some stage boundary; used for matched-group baselining.
  - compute_stage_potentials: Phi_i = alpha*m_i + beta*r_i - gamma*c_i.

Everything here is pure-python (no torch / no model deps) so it can be imported
both by the rollout builder and by the credit-target / training scripts.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


# Allowed enum-like string values (kept as plain strings for JSONL friendliness).
BEFORE_AFTER_VALUES = ("sat", "unsat")
FAILURE_TYPES = ("none", "wrong_value", "missing", "format", "constraint_violation")


@dataclass
class StageRecord:
    """One verifier stage record (an element of Sigma_RL for a rollout).

    Attributes:
        predicate_id: which checkpoint / predicate from L_env this stage targets.
        before:       "sat" | "unsat"  -- state of this predicate before this stage.
        after:        "sat" | "unsat"  -- state of this predicate after this stage.
        failure_type: one of FAILURE_TYPES.
        char_span:    [start, end) char offsets into the completion text covered by
                      this stage's tokens. End is exclusive.
        info:         free-form dict, e.g. {"expected": ..., "got": ...}.
    """

    predicate_id: str
    before: str
    after: str
    failure_type: str
    char_span: Tuple[int, int]
    info: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # JSONL/parquet-friendly: char_span as a 2-element list of ints.
        s, e = self.char_span
        d["char_span"] = [int(s), int(e)]
        return d

    @staticmethod
    def from_dict(d: dict) -> "StageRecord":
        span = d.get("char_span") or (0, 0)
        s, e = int(span[0]), int(span[1])
        return StageRecord(
            predicate_id=str(d.get("predicate_id", "")),
            before=str(d.get("before", "unsat")),
            after=str(d.get("after", "unsat")),
            failure_type=str(d.get("failure_type", "none")),
            char_span=(s, e),
            info=dict(d.get("info", {}) or {}),
        )


class StageVerifier(abc.ABC):
    """Abstract base for a stage / checkpoint verifier.

    gold = {answer: str, checkpoints: [{id, gold, order}], ...}
    """

    @abc.abstractmethod
    def predicate_library(self, gold: dict) -> List[str]:
        """Return L_env for this problem: the ordered list of predicate_ids."""
        raise NotImplementedError

    @abc.abstractmethod
    def verify(self, completion: str, gold: dict) -> List[StageRecord]:
        """Return one StageRecord per gold checkpoint (in checkpoint order).

        char_span entries index into `completion`.
        """
        raise NotImplementedError


def sigma_key(records_up_to_i: List[StageRecord]) -> str:
    """Stable string for the frozenset of satisfied predicate_ids in the given prefix.

    CANONICAL CONVENTION (exclusive): callers pass the EXCLUSIVE prefix records[:i],
    i.e. the predicates satisfied ENTERING stage i (state BEFORE the action at stage i).
    This is the counterfactual matching unit; build_solver_grpo_rollouts, build_credit_targets,
    apply_credit_head and inject_stage_corruptions all pass records[:i].
    A predicate is "satisfied" in the prefix iff any record for that predicate_id
    has after == "sat". The key is a deterministic, order-independent string so it
    can be used to group same-state stages across rollouts (matched-group baseline).
    """
    satisfied = set()
    for rec in records_up_to_i:
        # Accept either StageRecord or dict (from JSONL).
        if isinstance(rec, StageRecord):
            after = rec.after
            pid = rec.predicate_id
        else:
            after = str(rec.get("after", "unsat"))
            pid = str(rec.get("predicate_id", ""))
        if after == "sat":
            satisfied.add(pid)
    # frozenset for set semantics, sorted for a stable, reproducible string.
    return "{" + ",".join(sorted(frozenset(satisfied))) + "}"


def compute_stage_potentials(
    records: List[StageRecord],
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.1,
    cost_per_stage: Optional[List[float]] = None,
) -> List[float]:
    """Stage potential Phi_i for each stage.

        Phi_i = alpha*m_i + beta*r_i - gamma*c_i

    where, scanning stages in order:
      m_i = 1 if this stage is the FIRST attainment of a previously-unsat
            admissible predicate (before == "unsat" and after == "sat"
            and that predicate has not yet been attained earlier).
      r_i = 1 if this stage is a RECOVERY: a predicate that FAILED at some earlier
            stage (a record with after == "unsat") becomes "sat" at stage i.
            (Window default = global, i.e. any earlier failure counts.)
      c_i = normalized cost = (char-length of this stage's span) / (sum of all
            stage span lengths). If `cost_per_stage` is provided it overrides this
            and is used directly (already normalized by the caller).

    Returns a list[float] aligned to `records`.
    """
    n = len(records)
    if n == 0:
        return []

    # --- normalized cost c_i ---
    if cost_per_stage is not None:
        if len(cost_per_stage) != n:
            raise ValueError(
                f"cost_per_stage length {len(cost_per_stage)} != n_records {n}"
            )
        costs = [float(c) for c in cost_per_stage]
    else:
        span_lens = []
        for rec in records:
            span = rec.char_span if isinstance(rec, StageRecord) else rec.get("char_span", (0, 0))
            s, e = int(span[0]), int(span[1])
            span_lens.append(max(0, e - s))
        total = float(sum(span_lens))
        if total > 0.0:
            costs = [ln / total for ln in span_lens]
        else:
            costs = [0.0] * n

    # --- m_i (first attainment) and r_i (recovery) ---
    attained: set = set()        # predicate_ids attained (after==sat) at any earlier stage
    failed_earlier: set = set()  # predicate_ids that recorded a failure (after==unsat) earlier

    potentials: List[float] = []
    for i, rec in enumerate(records):
        if isinstance(rec, StageRecord):
            before = rec.before
            after = rec.after
            pid = rec.predicate_id
        else:
            before = str(rec.get("before", "unsat"))
            after = str(rec.get("after", "unsat"))
            pid = str(rec.get("predicate_id", ""))

        m_i = 1.0 if (before == "unsat" and after == "sat" and pid not in attained) else 0.0
        # Recovery: predicate became sat here AND it had failed at an earlier stage.
        r_i = 1.0 if (after == "sat" and pid in failed_earlier) else 0.0

        phi = alpha * m_i + beta * r_i - gamma * float(costs[i])
        potentials.append(float(phi))

        # update running state AFTER scoring stage i
        if after == "sat":
            attained.add(pid)
        else:  # after == "unsat" -> records a failure that a later stage may recover
            failed_earlier.add(pid)

    return potentials
