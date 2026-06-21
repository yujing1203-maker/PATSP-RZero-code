# -*- coding: utf-8 -*-
"""LongReason one-step multiple-choice reasoning environment.

This adapts lz1bytedance/LongReason into the AgenticEnv interface used by
PATSP/R-Zero.

Environment contract:
  observation = LongReason prompt + valid option letters
  action      = answer option letter
  reward      = 1.0 if predicted letter == gold, else 0.0
  done        = True after one executor action

This is a one-step environment, not a multi-step interactive simulator.
It is intended to make LongReason usable as the fifth training benchmark
in the same rollout/training framework.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from envs.base import Task, Observation, Action, StepResult, AgenticEnv
from verifiers.base import StageRecord


_ANY_LETTERS = "ABCDEFG"


def _default_root() -> Path:
    return Path(os.environ.get("RZERO_STORAGE", str(Path.home() / "RZero_storage"))) / "longreason_train_eval"


def _train_jsonl() -> Path:
    return Path(os.environ.get(
        "LONGREASON_TRAIN_JSONL",
        str(_default_root() / "longreason_8k_train.jsonl"),
    ))


def _eval_jsonl() -> Path:
    return Path(os.environ.get(
        "LONGREASON_EVAL_JSONL",
        str(_default_root() / "longreason_8k_eval.jsonl"),
    ))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"LongReason JSONL not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"LongReason JSONL is empty: {path}")
    return rows


def _normalize_letters(xs: Any) -> List[str]:
    out: List[str] = []
    for x in xs or []:
        x = str(x).strip().upper()
        if len(x) == 1 and x in _ANY_LETTERS and x not in out:
            out.append(x)
    return out


def _format_allowed(letters: List[str]) -> str:
    if len(letters) == 1:
        return letters[0]
    if len(letters) == 2:
        return f"{letters[0]} or {letters[1]}"
    return ", ".join(letters[:-1]) + f", or {letters[-1]}"


def _parse_answer(raw: str) -> Optional[str]:
    """Parse model raw action text into an answer letter A-G.

    We accept exact one-letter outputs and explicit answer patterns.
    We intentionally avoid a broad single-letter fallback because reasoning text
    may contain variables A/B/C and cause false positives.
    """
    t = (raw or "").strip()
    if not t:
        return None

    # Strip common wrappers.
    m = re.search(r"<action>(.*?)</action>", t, flags=re.I | re.S)
    if m:
        t = m.group(1).strip()

    u = t.upper().strip()

    if u in set(_ANY_LETTERS):
        return u

    patterns = [
        rf"FINAL\s+ANSWER\s*[:：]?\s*([{_ANY_LETTERS}])\b",
        rf"ANSWER\s*[:：]?\s*([{_ANY_LETTERS}])\b",
        rf"THE\s+ANSWER\s+IS\s*([{_ANY_LETTERS}])\b",
        rf"OPTION\s*([{_ANY_LETTERS}])\b",
        rf"^\s*([{_ANY_LETTERS}])[\.\)\s]*$",
    ]

    for pat in patterns:
        mm = re.search(pat, u)
        if mm:
            return mm.group(1)

    return None


class LongReasonEnv(AgenticEnv):
    """One-step MCQ environment for LongReason."""

    def __init__(self, seed: int = 0):
        self._seed = int(seed)
        self._counter = 0

        self._train_path = _train_jsonl()
        self._eval_path = _eval_jsonl()

        self._train_rows: Optional[List[Dict[str, Any]]] = None
        self._eval_rows: Optional[List[Dict[str, Any]]] = None

        # episode state
        self._task: Optional[Task] = None
        self._prompt: str = ""
        self._answer: str = ""
        self._option_letters: List[str] = []
        self._done: bool = False
        self._last_reward: float = 0.0
        self._last_prediction: Optional[str] = None
        self._last_raw: str = ""

    # ------------------------------------------------------------------ loading
    def _rows_for_split(self, split: str) -> List[Dict[str, Any]]:
        sp = str(split or "train").lower()
        if sp in {"eval", "valid", "validation", "test", "d2"}:
            if self._eval_rows is None:
                self._eval_rows = _load_jsonl(self._eval_path)
            return self._eval_rows

        if self._train_rows is None:
            self._train_rows = _load_jsonl(self._train_path)
        return self._train_rows

    def _row_to_task(self, row: Dict[str, Any]) -> Task:
        exid = str(row["example_idx"])
        source_split = str(row.get("source_split", "8k"))
        option_letters = _normalize_letters(row.get("option_letters", []))
        answer = str(row["answer"]).strip().upper()

        if answer and answer not in option_letters:
            option_letters.append(answer)

        task_id = str(row.get("id") or f"longreason_{source_split}_{exid}")

        return Task(
            task_id=task_id,
            spec={
                "goal": "answer the LongReason multiple-choice question",
                "prompt": str(row["prompt"]),
            },
            constraints={
                "answer": answer,
                "option_letters": option_letters,
                "example_idx": exid,
                "source_split": source_split,
            },
            info={
                "env": "longreason",
                "dataset": row.get("dataset", "lz1bytedance/LongReason"),
                "input_tokens": int(row.get("input_tokens", 0) or 0),
            },
        )

    # ------------------------------------------------------------------ contract
    def sample_new_task(self, rng: Any, difficulty: float) -> Task:
        rows = self._rows_for_split("train")
        r = rng if isinstance(rng, random.Random) else random.Random(self._seed + self._counter)
        row = rows[r.randrange(len(rows))]
        self._counter += 1
        return self._row_to_task(row)

    def list_tasks(self, split: str) -> List[Task]:
        """Return tasks for evaluation splits only.

        For train, return [] so build_agentic_rollouts.py will call
        sample_new_task(rng, difficulty) and randomly sample from the
        LongReason train pool instead of always taking the first N rows.
        """
        sp = str(split or "train").lower()
        if sp in {"eval", "valid", "validation", "test", "d2"}:
            rows = self._rows_for_split(sp)
            return [self._row_to_task(r) for r in rows]
        return []

    def predicate_library(self, task: Task) -> List[str]:
        return ["answer_correct"]

    def is_well_posed(self, task: Task) -> bool:
        prompt = task.spec.get("prompt", "")
        ans = task.constraints.get("answer", "")
        opts = _normalize_letters(task.constraints.get("option_letters", []))
        return bool(prompt) and bool(ans) and ans in opts and len(opts) >= 2

    def reset(self, task: Task) -> Observation:
        self._task = task
        self._prompt = str(task.spec["prompt"])
        self._answer = str(task.constraints["answer"]).strip().upper()
        self._option_letters = _normalize_letters(task.constraints.get("option_letters", []))
        self._done = False
        self._last_reward = 0.0
        self._last_prediction = None
        self._last_raw = ""
        return self._make_obs()

    def _make_obs(self) -> Observation:
        allowed = _format_allowed(self._option_letters) if self._option_letters else "A, B, C, or D"

        lines = [
            "LONGREASON MULTIPLE-CHOICE TASK",
            "",
            "Read the question and choose the correct option.",
            f"Valid option letters: {allowed}.",
            "",
            "QUESTION:",
            self._prompt,
            "",
            f"Action format: output exactly one uppercase letter from {allowed}.",
            "Do not explain. Do not output words. Do not use punctuation.",
        ]

        state = {
            "example_idx": self._task.constraints.get("example_idx") if self._task else None,
            "source_split": self._task.constraints.get("source_split") if self._task else None,
            "option_letters": list(self._option_letters),
            "done": bool(self._done),
            "last_prediction": self._last_prediction,
        }

        return Observation(
            text="\n".join(lines),
            available_actions=list(self._option_letters),
            state=state,
        )

    def step(self, action: Action) -> StepResult:
        if self._task is None:
            raise RuntimeError("step() before reset()")

        if self._done:
            return StepResult(
                obs=self._make_obs(),
                stage_records=[],
                done=True,
                terminal_reward=self._last_reward,
                info={"already_done": True},
            )

        raw = action.raw or ""
        pred = _parse_answer(raw)

        self._last_raw = raw
        self._last_prediction = pred
        self._done = True

        info: Dict[str, Any] = {
            "raw_action": raw,
            "prediction": pred,
            "gold": self._answer,
            "option_letters": list(self._option_letters),
        }

        before = "unsat"
        after = "unsat"
        failure_type = "wrong"

        if pred is None:
            reward = 0.0
            failure_type = "format"
            info["failure_type"] = "format"
            info["msg"] = "could not parse a final answer letter"
        elif pred not in self._option_letters:
            reward = 0.0
            failure_type = "constraint_violation"
            info["failure_type"] = "constraint_violation"
            info["msg"] = f"predicted option {pred!r} not in valid options {self._option_letters}"
        elif pred == self._answer:
            reward = 1.0
            after = "sat"
            failure_type = "none"
            info["failure_type"] = "none"
        else:
            reward = 0.0
            failure_type = "wrong"
            info["failure_type"] = "wrong"

        self._last_reward = float(reward)

        rec = StageRecord(
            predicate_id="answer_correct",
            before=before,
            after=after,
            failure_type=failure_type,
            char_span=(0, 0),
            info=info,
        )

        return StepResult(
            obs=self._make_obs(),
            stage_records=[rec],
            done=True,
            terminal_reward=float(reward),
            info=info,
        )

    def verify_final(self) -> float:
        return float(self._last_reward)

    def needs_replan(self, last: StepResult) -> bool:
        return False
