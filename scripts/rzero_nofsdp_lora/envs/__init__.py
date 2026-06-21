# -*- coding: utf-8 -*-
"""
Agentic environments subpackage.

Public surface:
  - Task, Observation, Action, StepResult : FROZEN dataclasses (envs/base.py).
  - AgenticEnv                            : ABC the rollout loop / challenger consume.

Concrete benchmark envs (textcraft / scienceworld / alfworld / herobench) are imported
directly from their modules by the two entry scripts; this package import stays heavy-
dependency-free (no torch / no vllm / no java).
"""

from .base import (
    Task,
    Observation,
    Action,
    StepResult,
    AgenticEnv,
)

__all__ = [
    "Task",
    "Observation",
    "Action",
    "StepResult",
    "AgenticEnv",
]
