# -*- coding: utf-8 -*-
"""
Agentic env contracts (frozen contract).

These dataclasses + the AgenticEnv ABC are the contract that the multi-turn
rollout loop (rollout_agentic.py), the challenger (generate_agentic_challenger_tasks.py)
and every concrete env (textcraft / scienceworld / alfworld / herobench) share.

REUSE: StageRecord comes from verifiers.base (NOT redefined here). StepResult.stage_records
is a list[StageRecord]; predicate_ids MUST come from env.predicate_library(task).
All credit-bearing signal is verifier-grounded; NO free-text reward.

Pure-python, no torch / no vllm: importable on a CPU box for the dry-run path.
"""

from __future__ import annotations

import abc
import os
import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional

# Robust import so `from envs.base import ...` and direct execution both work.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verifiers.base import StageRecord  # noqa: E402  (REUSED unchanged)


@dataclass
class Task:
    """A machine-checkable agentic task.

    spec:        problem statement / instructions (what the Planner+Executor read).
    constraints: machine-checkable constraints (the verifier grounds predicates on these).
    info:        free-form metadata (difficulty, split, provenance, ...).
    """

    task_id: str
    spec: dict
    constraints: dict
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "spec": dict(self.spec),
            "constraints": dict(self.constraints),
            "info": dict(self.info),
        }

    @staticmethod
    def from_dict(d: dict) -> "Task":
        return Task(
            task_id=str(d.get("task_id", "")),
            spec=dict(d.get("spec", {}) or {}),
            constraints=dict(d.get("constraints", {}) or {}),
            info=dict(d.get("info", {}) or {}),
        )


@dataclass
class Observation:
    """What the Executor sees this turn.

    text:              natural-language rendering of the current state.
    available_actions: action-space hints (e.g. ["add <i>", "remove <i>", "finalize"]).
    state:             structured state dict the env maintains internally.
    """

    text: str
    available_actions: List[str] = field(default_factory=list)
    state: dict = field(default_factory=dict)


@dataclass
class Action:
    """A parsed Executor action.

    raw:    the verbatim text region the Executor emitted (inside <action>...</action>).
    parsed: env-specific structured form, e.g. {"op": "add", "item": 3}.
    """

    raw: str
    parsed: dict = field(default_factory=dict)


@dataclass
class StepResult:
    """Result of advancing the env by one Executor action.

    stage_records: predicates whose state CHANGED this step (verifier-grounded),
                   using verifiers.base.StageRecord. char_span here is a PLACEHOLDER
                   (the env does not know the executor completion text); the rollout
                   loop fills char_span INTO the executor turn completion.
    done:          episode terminates after this step.
    terminal_reward: final verifier success in [0,1] when known at step time, else None.
    info:          free-form (e.g. {"needs_replan": bool, "failure_type": ...}).
    """

    obs: Observation
    stage_records: List[StageRecord] = field(default_factory=list)
    done: bool = False
    terminal_reward: Optional[float] = None
    info: dict = field(default_factory=dict)


class AgenticEnv(abc.ABC):
    """Abstract agentic environment (FROZEN signatures)."""

    @abc.abstractmethod
    def list_tasks(self, split: str) -> List[Task]:
        """Return the tasks for a named split (e.g. 'train' / 'eval')."""
        raise NotImplementedError

    @abc.abstractmethod
    def predicate_library(self, task: Task) -> List[str]:
        """L_env for this task: the ordered list of predicate_ids (e.g. 21 checkpoint ids)."""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, task: Task) -> Observation:
        """Reset to the start state for `task`; return the initial Observation."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action: Action) -> StepResult:
        """Advance one Executor action; return StepResult (changed-predicate stage_records)."""
        raise NotImplementedError

    @abc.abstractmethod
    def verify_final(self) -> float:
        """Terminal verifier success in [0,1] (programmatic, never free-text)."""
        raise NotImplementedError

    @abc.abstractmethod
    def needs_replan(self, last: StepResult) -> bool:
        """Verifier-EVENT replan trigger (failure_type != none / checkpoint regress)."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_new_task(self, rng: Any, difficulty: float) -> Task:
        """Challenger: generate a genuinely NEW machine-checkable task (not a resample)."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_well_posed(self, task: Task) -> bool:
        """Challenger gate: the task is solvable AND verifiable."""
        raise NotImplementedError
