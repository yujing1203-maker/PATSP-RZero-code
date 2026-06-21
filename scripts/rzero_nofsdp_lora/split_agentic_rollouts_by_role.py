# -*- coding: utf-8 -*-
"""
Role-split agentic trajectories into the EXISTING per-row schema.

Reads agentic_trajectories.jsonl (one row per (task_id,
sample) = {task_id, sample_index, terminal_reward, turns[], predicate_library}) and emits
TWO files:

  - planner_rollouts.jsonl  : one row per PLANNER turn.
  - executor_rollouts.jsonl : one row per EXECUTOR turn.

EACH output row is EXACTLY the existing per-row schema, so the EXISTING
machinery (build_credit_targets.py / train_credit_head.py / apply_credit_head.py /
train_lora_grpo_clip_from_rollouts.py --credit_mode) consumes the files VERBATIM. For each turn:

    uid                  = task_id
    sample_index         = trajectory sample_index
    turn_uid             = f"{task_id}::{sample_index}::t{turn_index}"   (P1: EVERY row)
    prompt               = turn.prompt
    completion           = turn.completion
    completion_for_train = completion
    stage_records        = turn.stage_records (char_span indexes INTO this completion)
    stage_potentials     = compute_stage_potentials(records)          [verifiers.base, REUSED]
    sigma_keys           = [sigma_key(records[:i]) for i in range(n)]  EXCLUSIVE prefix
    outcome_advantage    = group-relative over SAME-task rollouts of terminal_reward
                           ((r - mean)/(std + 1e-6); the build_solver_grpo_rollouts
                            compute_group_advantages pattern)
    advantage            = outcome_advantage   (scalar, for uniform credit_mode)
    stage_advantages     = EXECUTOR rows: None (filled later by apply_credit_head.py).
                           PLANNER rows (P3): one value per subgoal record, broadcasting
                           the plan-segment planner advantage A_P. A_P is the group-relative
                           (over SAME-task rollouts) of the SUM of executor-achieved Phi_i
                           in the segment that plan governed (governed_turn_indices).

The remaining per-row fields the trainer references (problem/answer/pred_answer/reward/
source/verification/note/solver_lora) are filled with agentic-appropriate values so the row
is a drop-in for the existing schema: `reward` = the trajectory terminal_reward (so any
re-derivation of group-relative advantage on `reward` matches outcome_advantage), `source`
= "agentic_<role>", and the role/turn provenance is kept under `note` + explicit fields.

REUSE, do not reimplement: StageRecord, sigma_key, compute_stage_potentials from
verifiers.base. EXCLUSIVE sigma [:i] everywhere (matches build_solver_grpo_rollouts.py,
build_credit_targets.py, apply_credit_head.py).

Pure python: NO torch, NO vllm. Runs on the mock --dry_run trajectories on a CPU box.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Robust import ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verifiers.base import (  # noqa: E402  (REUSED UNCHANGED)
    StageRecord,
    sigma_key,
    compute_stage_potentials,
)


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split agentic_trajectories.jsonl into per-role rollout files in the "
        "EXISTING per-row schema.",
    )
    p.add_argument(
        "--trajectories_jsonl",
        required=True,
        help="Input agentic_trajectories.jsonl rows.",
    )
    p.add_argument(
        "--out_planner",
        required=True,
        help="Output planner_rollouts.jsonl (one row per planner turn).",
    )
    p.add_argument(
        "--out_executor",
        required=True,
        help="Output executor_rollouts.jsonl (one row per executor turn).",
    )
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Group-relative advantage on terminal_reward (REUSED PATTERN)                  #
# --------------------------------------------------------------------------- #
def compute_group_advantages_over_terminal(
    trajectories: List[dict],
) -> Dict[tuple, float]:
    """Group-relative advantage of terminal_reward over SAME-task rollouts.

    This is the build_solver_grpo_rollouts.compute_group_advantages pattern, keyed by
    uid=task_id, operating on the per-trajectory terminal_reward:

        mean = mean(rewards in group)
        std  = sample std (denominator len-1)
        adv  = (reward - mean) / (std + 1e-6)        (0 if group size<=1 or std<1e-8)

    Returns {(task_id, sample_index): outcome_advantage}. One trajectory == one
    (task_id, sample_index); every turn from that trajectory inherits this scalar.
    """
    by_uid: Dict[str, List[int]] = defaultdict(list)
    for idx, traj in enumerate(trajectories):
        by_uid[str(traj.get("task_id", ""))].append(idx)

    adv_by_key: Dict[tuple, float] = {}
    for _uid, indices in by_uid.items():
        rewards = [float(trajectories[i].get("terminal_reward", 0.0)) for i in indices]

        # Default 0.0 for every member (covers group size<=1 and degenerate std).
        for i in indices:
            traj = trajectories[i]
            adv_by_key[(str(traj.get("task_id", "")), int(traj.get("sample_index", 0)))] = 0.0

        if len(rewards) <= 1:
            continue
        mean = sum(rewards) / len(rewards)
        var_num = sum((x - mean) ** 2 for x in rewards)
        std = math.sqrt(var_num / (len(rewards) - 1)) if len(rewards) > 1 else 0.0
        if std < 1e-8:
            continue
        for i in indices:
            traj = trajectories[i]
            key = (str(traj.get("task_id", "")), int(traj.get("sample_index", 0)))
            adv_by_key[key] = (float(traj.get("terminal_reward", 0.0)) - mean) / (std + 1e-6)

    return adv_by_key


# --------------------------------------------------------------------------- #
# Planner credit A_P (P3): group-relative SUM of governed executor Phi_i        #
# --------------------------------------------------------------------------- #
def _executor_phi_sum_by_turn_index(traj: dict) -> Dict[int, float]:
    """Map each EXECUTOR turn_index -> SUM of that turn's achieved stage potentials Phi_i.

    Phi_i is computed by the SAME REUSED compute_stage_potentials over the executor turn's
    own stage_records, so the planner-segment value is the sum of exactly the executor Phi
    that build_role_row writes onto the executor rows.
    """
    phi_sum: Dict[int, float] = {}
    for turn in traj.get("turns", []) or []:
        if str(turn.get("role", "")) != "executor":
            continue
        raw_records = turn.get("stage_records", []) or []
        records = [StageRecord.from_dict(r) for r in raw_records]
        potentials = compute_stage_potentials(records)
        ti = int(turn.get("turn_index", 0))
        phi_sum[ti] = phi_sum.get(ti, 0.0) + float(sum(float(p) for p in potentials))
    return phi_sum


def _governed_turn_indices(traj: dict, planner_turn: dict) -> List[int]:
    """Executor turn_index values that this planner turn governs (P4 governed segment).

    Prefers an explicit `governed_turn_indices` on the PlannerTurn (sibling rollout_agentic
    fix). Defensive fallback (field absent): all EXECUTOR turn_index values from this plan's
    turn_index (exclusive) until the next PLANNER turn (or episode end), i.e. the segment
    this plan governs until the next replan/terminal.
    """
    explicit = planner_turn.get("governed_turn_indices")
    if explicit is not None:
        out: List[int] = []
        for v in explicit:
            try:
                out.append(int(v))
            except (TypeError, ValueError):
                continue
        return out

    # Fallback: scan turns ordered by turn_index; collect executor turns after this
    # planner boundary until the next planner turn.
    this_ti = int(planner_turn.get("turn_index", 0))
    turns = sorted(
        (traj.get("turns", []) or []),
        key=lambda t: int(t.get("turn_index", 0)),
    )
    governed: List[int] = []
    seen_self = False
    for t in turns:
        ti = int(t.get("turn_index", 0))
        role = str(t.get("role", ""))
        if not seen_self:
            if ti == this_ti and role == "planner":
                seen_self = True
            continue
        # after this planner turn
        if role == "planner":
            break  # next replan boundary ends this plan's governed segment
        if role == "executor":
            governed.append(ti)
    return governed


def compute_planner_advantages(
    trajectories: List[dict],
) -> Dict[tuple, float]:
    """Group-relative planner advantage A_P per planner turn (P3).

    For each planner turn we compute its plan-segment value = SUM of executor-achieved
    Phi_i over the turn's governed_turn_indices (P4). Then, REUSING the same group pattern
    as compute_group_advantages_over_terminal ((v - mean)/(std + 1e-6), denominator len-1,
    0 for group size<=1 or degenerate std), we standardize these segment values over the
    SAME-task rollouts (keyed by task_id).

    Returns {(task_id, sample_index, planner_turn_index): A_P}. The planner row broadcasts
    its A_P across all of its subgoal stage_records.
    """
    # Collect per-planner-turn segment values, grouped by task_id.
    by_task: Dict[str, List[tuple]] = defaultdict(list)  # task_id -> [(key, value)]
    for traj in trajectories:
        task_id = str(traj.get("task_id", ""))
        sample_index = int(traj.get("sample_index", 0))
        phi_by_ti = _executor_phi_sum_by_turn_index(traj)
        for turn in traj.get("turns", []) or []:
            if str(turn.get("role", "")) != "planner":
                continue
            governed = _governed_turn_indices(traj, turn)
            seg_value = float(sum(phi_by_ti.get(ti, 0.0) for ti in governed))
            key = (task_id, sample_index, int(turn.get("turn_index", 0)))
            by_task[task_id].append((key, seg_value))

    adv_by_key: Dict[tuple, float] = {}
    for _task_id, members in by_task.items():
        # Default 0.0 for every member (covers group size<=1 and degenerate std).
        for key, _v in members:
            adv_by_key[key] = 0.0
        values = [v for _k, v in members]
        if len(values) <= 1:
            continue
        mean = sum(values) / len(values)
        var_num = sum((x - mean) ** 2 for x in values)
        std = math.sqrt(var_num / (len(values) - 1)) if len(values) > 1 else 0.0
        if std < 1e-8:
            continue
        for key, v in members:
            adv_by_key[key] = (v - mean) / (std + 1e-6)

    return adv_by_key


# --------------------------------------------------------------------------- #
# Per-turn row construction                                                     #
# --------------------------------------------------------------------------- #
def _turn_field(turn: dict, name: str) -> Any:
    """Read a label field from a turn, preferring turn-level then turn.info.

    The agentic corruptor (inject_agentic_corruptions.py) records the causal label on BOTH
    the corrupted executor turn's `info` and as turn-level mirrors. We look turn-level first,
    then fall into turn.info, returning None when the field is absent on this turn.
    """
    if name in turn and turn.get(name) is not None:
        return turn.get(name)
    info = turn.get("info") or {}
    if isinstance(info, dict) and info.get(name) is not None:
        return info.get(name)
    return None


def build_role_row(
    traj: dict,
    turn: dict,
    outcome_advantage: float,
    planner_advantage: Optional[float] = None,
) -> Dict[str, Any]:
    """Project one Turn onto the EXISTING per-row schema.

    stage_records are deserialized to StageRecord then re-serialized via to_dict() so the
    char_span list / failure_type / info shape matches what the credit machinery expects
    (identical to build_solver_grpo_rollouts.py output). compute_stage_potentials + the
    EXCLUSIVE sigma_keys are computed from the SAME StageRecord list.
    """
    completion = str(turn.get("completion", ""))
    prompt = str(turn.get("prompt", ""))

    role = str(turn.get("role", ""))

    # --- stage_records (REUSE StageRecord) ---
    raw_records = turn.get("stage_records", []) or []
    records: List[StageRecord] = [StageRecord.from_dict(r) for r in raw_records]

    # --- INTERVENTIONAL label copy (PINNED contract) -----------------------------------
    # For EXECUTOR rows, COPY the corruption label off the turn VERBATIM (turn-level then
    # turn.info), defaulting is_intervention=False. The agentic/stage corruptors record
    # is_intervention / corrupted_stage_index / downstream_success_drop on the corrupted
    # turn so build_credit_targets' interventional branch can read them straight off the
    # post-split executor row. Planner rows are NEVER intervened on (P2/P3) -> skip them.
    is_intervention = False
    corrupted_stage_index: Optional[int] = None
    downstream_success_drop: Optional[float] = None
    if role == "executor":
        raw_is_interv = _turn_field(turn, "is_intervention")
        is_intervention = bool(raw_is_interv) if raw_is_interv is not None else False
        if is_intervention:
            csi = _turn_field(turn, "corrupted_stage_index")
            try:
                corrupted_stage_index = int(csi) if csi is not None else None
            except (TypeError, ValueError):
                corrupted_stage_index = None
            dsd = _turn_field(turn, "downstream_success_drop")
            try:
                downstream_success_drop = (
                    float(dsd) if dsd is not None else 0.0
                )
            except (TypeError, ValueError):
                downstream_success_drop = 0.0

            # PINNED: the interventional row must NEVER be empty and corrupted_stage_index
            # must index INTO THIS ROW's stage_records. If the corrupted action produced no
            # stage_record (changed no predicate / invalid action), SYNTHESIZE one for the
            # targeted predicate (after="unsat", failure_type set) so the row is consumable.
            if not records:
                targeted_pid = str(
                    _turn_field(turn, "predicate_id")
                    or _turn_field(turn, "corruption_mode")
                    or "corrupted_predicate"
                )
                failure_type = str(_turn_field(turn, "failure_type") or "intervention")
                synth_span = (0, max(0, len(completion)))
                records = [
                    StageRecord(
                        predicate_id=targeted_pid,
                        before="sat",
                        after="unsat",
                        failure_type=failure_type,
                        char_span=synth_span,
                        info={"synthesized_for_intervention": True},
                    )
                ]
            # Clamp corrupted_stage_index into THIS ROW's stage_records range (a single
            # synthesized record collapses any out-of-range / turn-ordinal value to 0).
            if corrupted_stage_index is None or not (
                0 <= corrupted_stage_index < len(records)
            ):
                corrupted_stage_index = max(0, min(corrupted_stage_index or 0, len(records) - 1))

    # --- stage_potentials Phi_i (REUSE compute_stage_potentials) ---
    # char_span here indexes into THIS completion, so the default cost = span-length /
    # total-span-length normalization in compute_stage_potentials is correct;
    # we let it derive costs from the records' char_spans (no override).
    potentials = compute_stage_potentials(records)

    # --- sigma_keys: EXCLUSIVE prefix records[:i] (PINNED convention) ---
    sigma_keys = [sigma_key(records[:i]) for i in range(len(records))]

    terminal_reward = float(traj.get("terminal_reward", 0.0))

    # P1: turn_uid identifies THIS role turn uniquely across the multi-turn episode so the
    # downstream hidden store (build_credit_targets) never collides planner/executor turns
    # that share (task_id, sample_index).
    task_id = str(traj.get("task_id", ""))
    sample_index = int(traj.get("sample_index", 0))
    turn_index = int(turn.get("turn_index", 0))
    turn_uid = f"{task_id}::{sample_index}::t{turn_index}"

    # P3: PLANNER rows ship non-null stage_advantages = the plan-segment A_P broadcast over
    # every subgoal stage_record. EXECUTOR rows keep None (apply_credit_head fills them).
    if role == "planner":
        stage_advantages: Optional[List[float]] = [
            float(planner_advantage or 0.0) for _ in range(len(records))
        ]
    else:
        stage_advantages = None

    row: Dict[str, Any] = {
        # --- existing per-row fields (KEPT) ---
        "uid": str(traj.get("task_id", "")),
        "sample_index": int(traj.get("sample_index", 0)),
        "turn_uid": turn_uid,
        "prompt": prompt,
        "completion": completion,
        "completion_for_train": completion,
        # No textual problem/answer in the agentic setting; carry context-bearing values.
        "problem": prompt,
        "answer": "",
        "pred_answer": "",
        # reward = the trajectory terminal_reward, so any downstream re-derivation of
        # group-relative advantage on `reward` reproduces outcome_advantage exactly.
        "reward": terminal_reward,
        "advantage": float(outcome_advantage),
        "source": f"agentic_{role}" if role else "agentic",
        "verification": "agentic_verifier_grounded",
        "note": (
            f"agentic role={role} boundary={turn.get('boundary')} "
            f"turn_index={turn.get('turn_index')}"
        ),
        "solver_lora": "",
        # --- new per-row fields ---
        "stage_records": [rec.to_dict() for rec in records],
        "stage_potentials": [float(p) for p in potentials],
        "outcome_advantage": float(outcome_advantage),
        # P3: planner rows non-null (broadcast A_P); executor rows None (apply_credit_head fills).
        "stage_advantages": stage_advantages,
        "sigma_keys": list(sigma_keys),
        # --- agentic provenance (extra; ignored by the trainer/credit machinery) ---
        "role": role,
        "boundary": int(turn.get("boundary", 0)),
        "turn_index": int(turn.get("turn_index", 0)),
        "terminal_reward": terminal_reward,
        # --- PINNED interventional contract (read by build_credit_targets K branch) ---
        # is_intervention defaults False (observational); the corrupted EXECUTOR row carries
        # the causal label VERBATIM so build_credit_targets emits a K target at
        # corrupted_stage_index with target=downstream_success_drop.
        "is_intervention": bool(is_intervention),
    }
    if is_intervention:
        row["corrupted_stage_index"] = int(corrupted_stage_index)
        row["downstream_success_drop"] = float(downstream_success_drop or 0.0)
    return row


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    in_path = Path(args.trajectories_jsonl)
    if not in_path.is_file():
        raise FileNotFoundError(f"--trajectories_jsonl not found: {in_path}")

    out_planner = Path(args.out_planner)
    out_executor = Path(args.out_executor)
    out_planner.parent.mkdir(parents=True, exist_ok=True)
    out_executor.parent.mkdir(parents=True, exist_ok=True)

    # Load all trajectories first (we need the full per-task group to compute the
    # group-relative outcome_advantage before emitting any row).
    trajectories: List[dict] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trajectories.append(json.loads(line))

    adv_by_key = compute_group_advantages_over_terminal(trajectories)
    # P3: per-planner-turn group-relative advantage A_P (governed executor Phi sum).
    planner_adv_by_key = compute_planner_advantages(trajectories)

    n_planner = 0
    n_executor = 0
    n_other = 0

    with out_planner.open("w", encoding="utf-8") as fp, \
            out_executor.open("w", encoding="utf-8") as fe:
        for traj in trajectories:
            key = (str(traj.get("task_id", "")), int(traj.get("sample_index", 0)))
            outcome_advantage = adv_by_key.get(key, 0.0)
            for turn in traj.get("turns", []) or []:
                role = str(turn.get("role", ""))
                planner_advantage = None
                if role == "planner":
                    pkey = (
                        str(traj.get("task_id", "")),
                        int(traj.get("sample_index", 0)),
                        int(turn.get("turn_index", 0)),
                    )
                    planner_advantage = planner_adv_by_key.get(pkey, 0.0)
                row = build_role_row(traj, turn, outcome_advantage, planner_advantage)
                if role == "planner":
                    fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_planner += 1
                elif role == "executor":
                    fe.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_executor += 1
                else:
                    # Unknown role: skip but count (keeps the split lossless to report).
                    n_other += 1

    summary = {
        "trajectories_jsonl": str(in_path),
        "out_planner": str(out_planner),
        "out_executor": str(out_executor),
        "n_trajectories": len(trajectories),
        "n_planner_rows": n_planner,
        "n_executor_rows": n_executor,
        "n_skipped_unknown_role": n_other,
        "sigma_convention": "EXCLUSIVE prefix records[:i]",
        "turn_uid_convention": "f'{task_id}::{sample_index}::t{turn_index}' on EVERY row (P1)",
        "outcome_advantage": "group-relative over same task_id of terminal_reward "
        "((r-mean)/(std+1e-6))",
        "planner_stage_advantages": "group-relative over same task_id of SUM of governed "
        "executor Phi_i ((v-mean)/(std+1e-6)); broadcast over the planner row's subgoals (P3)",
    }
    summary_path = out_planner.with_name("split_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[split_agentic_rollouts_by_role] planner_rows={n_planner} "
          f"executor_rows={n_executor} (skipped_unknown={n_other})")
    print(f"[split_agentic_rollouts_by_role] -> {out_planner}")
    print(f"[split_agentic_rollouts_by_role] -> {out_executor}")
    print(f"[split_agentic_rollouts_by_role] summary -> {summary_path}")


if __name__ == "__main__":
    main()
