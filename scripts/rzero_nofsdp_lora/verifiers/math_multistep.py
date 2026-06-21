# -*- coding: utf-8 -*-
"""
Multi-checkpoint MATH verifier.

Contract (math_multistep.verify):
  - predicate_library(gold): L_env from gold["checkpoints"].
  - verify(completion, gold):
      1. Extract the solver's ordered \\boxed{...} values from the completion,
         tracking the char span of each boxed value (the [start, end) of the
         "...\\boxed{<value>}" token region, end exclusive).
         Boxed-value extraction reuses mathruler.extract_boxed_content on each
         isolated "\\boxed{...}" substring, matching the existing import/usage in
         build_solver_grpo_rollouts.py / judge_candidates_with_solver.py.
      2. Greedily align the extracted boxed values IN ORDER to the ordered gold
         checkpoints, matching with mathruler.grade_answer.
      3. Emit one StageRecord per gold checkpoint:
            after = "sat" if a boxed value matched, else "unsat"
            failure_type in {none, wrong_value, missing}
            char_span = the matched boxed value's char span (or a zero-width span
            at the end of the last consumed region when missing).
  - milestone_recall(...): honest batch-level milestone recall helper
    (expect ~0.6, per MiRA -- alternate valid paths exist; this is a known,
    reported limitation, NOT hidden).

This module reuses existing helpers and does NOT reimplement boxed extraction or
answer grading.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

# Robust import for the verifiers subpackage when run as a loose module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verifiers.base import StageRecord, StageVerifier  # noqa: E402

# Same mathruler import path as build_solver_grpo_rollouts.py /
# judge_candidates_with_solver.py. extract_boxed_content returns the boxed value
# of a single "\\boxed{...}" substring; grade_answer does symbolic/numeric match.
from mathruler.grader import extract_boxed_content, grade_answer  # noqa: E402


def _find_boxed_spans(text: str) -> List[Tuple[str, int, int]]:
    r"""Find every ``\boxed{...}`` occurrence in ``text``, in order.

    Returns a list of (value, start, end) where:
      - value is the content extracted by mathruler.extract_boxed_content on the
        isolated ``\boxed{...}`` substring (so extraction semantics stay identical
        to the rest of the codebase),
      - [start, end) is the char span of the full ``\boxed{...}`` region in ``text``
        (start at the backslash of ``\boxed``, end exclusive after the closing ``}``).

    Brace matching is depth-aware so nested ``{}`` inside the boxed argument are
    handled (e.g. ``\boxed{\frac{1}{2}}``).
    """
    spans: List[Tuple[str, int, int]] = []
    marker = r"\boxed"
    i = 0
    n = len(text)
    while True:
        idx = text.find(marker, i)
        if idx < 0:
            break
        # locate the opening brace after \boxed (allow optional whitespace)
        j = idx + len(marker)
        while j < n and text[j] in " \t":
            j += 1
        if j >= n or text[j] != "{":
            # not a real \boxed{...}; skip past this marker
            i = idx + len(marker)
            continue
        # depth-aware scan for the matching closing brace
        depth = 0
        k = j
        end = -1
        while k < n:
            ch = text[k]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = k + 1  # exclusive
                    break
            k += 1
        if end < 0:
            # unbalanced; stop scanning to avoid an infinite loop
            break
        substring = text[idx:end]
        value = extract_boxed_content(substring)
        # mathruler returns "None" (string) when nothing is found; normalize.
        if value is None or value == "None":
            value = ""
        spans.append((value.strip(), idx, end))
        i = end
    return spans


class MathMultiStepVerifier(StageVerifier):
    """Ordered multi-checkpoint math verifier."""

    def __init__(self, grade_timeout: Optional[float] = None):
        # grade_timeout kept for signature parity with the rest of the codebase;
        # mathruler.grade_answer here is called directly (timeout handled by callers
        # that wrap it, e.g. judge_candidates_with_solver.grade_answer_timeout).
        self.grade_timeout = grade_timeout

    # ------------------------------------------------------------------ L_env
    def predicate_library(self, gold: dict) -> List[str]:
        checkpoints = _sorted_checkpoints(gold)
        return [str(cp["id"]) for cp in checkpoints]

    # ------------------------------------------------------------- grading
    @staticmethod
    def _match(pred: str, gold_value: str) -> bool:
        pred = (pred or "").strip()
        gold_value = (gold_value or "").strip()
        if not pred or not gold_value:
            return False
        if pred == gold_value:
            return True
        try:
            if grade_answer(pred, gold_value):
                return True
        except Exception:
            pass
        try:
            if grade_answer(gold_value, pred):
                return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------- verify
    def verify(self, completion: str, gold: dict) -> List[StageRecord]:
        completion = completion or ""
        checkpoints = _sorted_checkpoints(gold)
        n_ck = len(checkpoints)

        boxed = _find_boxed_spans(completion)  # list of (value, start, end), in order

        records: List[StageRecord] = []
        attained_pids: set = set()
        bi = 0            # pointer into boxed values (greedy, in order)
        last_end = 0      # char offset just past the last consumed boxed region

        for cp in checkpoints:
            pid = str(cp["id"])
            gold_value = str(cp.get("gold", ""))
            before = "sat" if pid in attained_pids else "unsat"

            matched_idx = None
            # Greedy in-order alignment: scan remaining boxed values for the first
            # that grades equal to this checkpoint's gold. Boxed values skipped here
            # are "wrong" intermediate values for earlier/other checkpoints and are
            # left available only for nothing earlier (in-order => we advance bi).
            scan = bi
            while scan < len(boxed):
                val, _s, _e = boxed[scan]
                if self._match(val, gold_value):
                    matched_idx = scan
                    break
                scan += 1

            if matched_idx is not None:
                val, s, e = boxed[matched_idx]
                after = "sat"
                failure_type = "none"
                char_span = (int(s), int(e))
                info = {"expected": gold_value, "got": val, "order": int(cp.get("order", 0))}
                attained_pids.add(pid)
                bi = matched_idx + 1
                last_end = int(e)
            else:
                # No remaining boxed value matches this checkpoint.
                after = "unsat"
                if bi < len(boxed):
                    # A boxed value is present at this position but does not match
                    # -> wrong value. Consume it (advance bi) so it is attributed to
                    # exactly one checkpoint and later checkpoints can still be
                    # classified as "missing" once the boxed values run out.
                    val, s, e = boxed[bi]
                    failure_type = "wrong_value"
                    char_span = (int(s), int(e))
                    info = {"expected": gold_value, "got": val, "order": int(cp.get("order", 0))}
                    bi += 1
                    last_end = int(e)
                else:
                    # No boxed value left for this checkpoint -> missing.
                    failure_type = "missing"
                    char_span = (int(last_end), int(last_end))  # zero-width span
                    info = {"expected": gold_value, "got": "", "order": int(cp.get("order", 0))}

            records.append(
                StageRecord(
                    predicate_id=pid,
                    before=before,
                    after=after,
                    failure_type=failure_type,
                    char_span=char_span,
                    info=info,
                )
            )

        # If there are no checkpoints at all, return empty (caller falls back to
        # uniform / scalar behavior, preserving backward compatibility).
        if n_ck == 0:
            return []

        return records


def _sorted_checkpoints(gold: dict) -> List[dict]:
    """Return gold checkpoints sorted by their `order` field (stable).

    Tolerates missing / garbled `order` by falling back to the original index.
    """
    checkpoints = list(gold.get("checkpoints") or [])
    ordered = sorted(
        enumerate(checkpoints),
        key=lambda t: (_safe_int(t[1].get("order"), t[0]), t[0]),
    )
    return [cp for _, cp in ordered]


def _safe_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def milestone_recall(
    batch_records: List[List[StageRecord]],
    per_problem: bool = False,
):
    """Honest milestone recall over a batch of rollouts.

    milestone recall = (# checkpoints reached, after == "sat")
                       / (# total checkpoints across the batch)

    Args:
        batch_records: list (one entry per rollout) of stage-record lists, each as
            returned by MathMultiStepVerifier.verify (StageRecord or dict).
        per_problem: if True, also return the list of per-rollout recall values.

    Returns:
        if per_problem is False: float overall recall in [0, 1].
        if per_problem is True:  (overall: float, per_rollout: list[float]).

    NOTE: ~0.6 is the expected ballpark (MiRA): alternate valid solution paths
    legitimately skip or reorder annotated checkpoints. We report this honestly
    rather than hide the limitation.
    """
    total = 0
    hit = 0
    per_rollout: List[float] = []
    for records in batch_records:
        r_total = 0
        r_hit = 0
        for rec in records:
            after = rec.after if isinstance(rec, StageRecord) else str(rec.get("after", "unsat"))
            r_total += 1
            if after == "sat":
                r_hit += 1
        total += r_total
        hit += r_hit
        per_rollout.append((r_hit / r_total) if r_total > 0 else 0.0)

    overall = (hit / total) if total > 0 else 0.0
    if per_problem:
        return overall, per_rollout
    return overall
