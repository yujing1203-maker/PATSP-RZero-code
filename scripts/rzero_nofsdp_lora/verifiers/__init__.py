# -*- coding: utf-8 -*-
"""
Stage verifier subpackage.

Public surface:
  - StageRecord            : per-stage verifier record (Sigma_RL), char_span indexes into `completion`.
  - StageVerifier          : ABC -> predicate_library(gold), verify(completion, gold).
  - compute_stage_potentials : Phi_i = alpha*m_i + beta*r_i - gamma*c_i.
  - sigma_key              : stable frozenset string of satisfied predicate_ids up to stage i.
  - MathMultiStepVerifier  : ordered-boxed-value multi-checkpoint math verifier.

Hard rule: everything here is verifier-grounded (numeric / predicate checks). Never reward free text.
"""

from .base import (
    StageRecord,
    StageVerifier,
    compute_stage_potentials,
    sigma_key,
)

# .math_multistep pulls in `mathruler`, which is an optional heavy dependency.
# base-only consumers (apply_credit_head, build_credit_targets, inject_stage_corruptions,
# the trainer) must import this package cleanly even when mathruler is absent.
# So we make the math_multistep re-export lazy: try eagerly, and on ImportError
# leave the names as None and fall back to PEP 562 __getattr__ for first-access import.
try:
    from .math_multistep import (
        MathMultiStepVerifier,
        milestone_recall,
    )
except ImportError:
    MathMultiStepVerifier = None
    milestone_recall = None


def __getattr__(name):
    # PEP 562: only invoked for attributes not found in the module globals.
    # If the eager import above failed, the names are bound to None and this is
    # not reached for them. This handles the case where the names were deleted
    # or to surface the original ImportError on explicit access.
    if name in ("MathMultiStepVerifier", "milestone_recall"):
        from . import math_multistep  # re-raises the real ImportError if mathruler missing
        return getattr(math_multistep, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "StageRecord",
    "StageVerifier",
    "compute_stage_potentials",
    "sigma_key",
    "MathMultiStepVerifier",
    "milestone_recall",
]
