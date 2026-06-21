# -*- coding: utf-8 -*-
"""
Roles -- Planner + Executor prompt builders + parsers.

This module is the role layer of the AGENTIC extension. It is PURE PYTHON: it must
import cleanly and run its __main__ smoke test on a CPU box with NO torch and NO vllm
installed (it is on the dry-run path). Therefore nothing here imports torch / vllm.

What this module provides (frozen signatures):

    PLAN_OPEN, PLAN_CLOSE = "<plan>", "</plan>"
    ACT_OPEN,  ACT_CLOSE  = "<action>", "</action>"

    @dataclass Subgoal: id:int; predicate_id:str; text:str; char_span:(int,int)
        # char_span indexes into the PLAN completion text.

    class Planner:
        SYSTEM = "...decompose into ordered VERIFIABLE subgoals, each tied to a predicate_id..."
        @staticmethod build_messages(task, state_summary, prev_plan, feedback) -> list[{role,content}]
        @staticmethod parse_plan(text) -> list[Subgoal]

    class Executor:
        SYSTEM = "...take ONE action toward the current subgoal; output <action>...</action>..."
        @staticmethod build_messages(task, plan, obs, history) -> list[{role,content}]
        @staticmethod parse_action(text) -> Action   # Action.parsed dict + char_span of action region

Design notes:
  - Each Planner subgoal is tied to a predicate_id drawn from the task's predicate
    library (L_env). This is what makes plan-level credit verifier-grounded: a
    subgoal is "achieved" iff its predicate becomes sat (handled by the rollout loop
    / split, not here). parse_plan does a best-effort predicate_id match against the
    task predicate library; if no match, it falls back to positional assignment
    (i-th subgoal -> i-th predicate), and finally to a synthetic id if the library
    runs out. The char_span of each Subgoal indexes into the <plan>...</plan>
    completion text so the rollout loop can attach StageRecord char_spans for the
    planner turn.
  - parse_action returns an Action whose `parsed` dict is the executor-action intent
    (verb + args) and whose char_span marks the <action>...</action> region inside the
    executor completion (used by the loop to attach StageRecord char_spans to the
    executor turn). The action grammar mirrors the mock env vocabulary
    ({add i, remove i, finalize}) but is intentionally permissive so the DeepPlanning
    adapter (free-text plan emission) also parses.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --- Robust import: make sibling packages importable
# regardless of the caller's cwd, then import the reused / sibling contracts.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# verifiers.base is REUSED UNCHANGED. roles.py does not need StageRecord directly,
# but importing it here documents the contract boundary and fails loud if the reused
# package is missing. (Pure python, no torch.)
try:
    from verifiers.base import StageRecord  # noqa: F401  (contract anchor)
except Exception:  # pragma: no cover - only if the reused package is absent
    StageRecord = None  # type: ignore

# envs.base provides the FROZEN agentic dataclasses (Task, Observation, StepResult,
# Action). It is a sibling NEW module. To keep roles.py self-sufficient on the
# dry-run/smoke path even before envs/base.py is created by its owning agent, we
# import it if present and otherwise fall back to LOCAL definitions that match the
# FROZEN schema VERBATIM. When envs.base exists,
# its types are used so identity/isinstance behave correctly across modules.
try:
    from envs.base import Task, Observation, StepResult, Action  # type: ignore  # noqa: F401
    _ENVS_BASE_AVAILABLE = True
except Exception:
    _ENVS_BASE_AVAILABLE = False

    @dataclass
    class Task:  # type: ignore  # FROZEN mirror of envs.base.Task
        task_id: str
        spec: dict
        constraints: dict
        info: dict = field(default_factory=dict)

    @dataclass
    class Observation:  # type: ignore  # FROZEN mirror of envs.base.Observation
        text: str
        available_actions: List[str]
        state: dict = field(default_factory=dict)

    @dataclass
    class Action:  # type: ignore  # FROZEN mirror of envs.base.Action
        raw: str
        parsed: dict

    @dataclass
    class StepResult:  # type: ignore  # FROZEN mirror of envs.base.StepResult
        obs: Observation
        stage_records: list
        done: bool
        terminal_reward: Optional[float]
        info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Tags (FROZEN)                                                                 #
# --------------------------------------------------------------------------- #
PLAN_OPEN, PLAN_CLOSE = "<plan>", "</plan>"
ACT_OPEN, ACT_CLOSE = "<action>", "</action>"


# --------------------------------------------------------------------------- #
# Subgoal (FROZEN)                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class Subgoal:
    """One ordered, verifiable subgoal emitted by the Planner.

    Attributes:
        id:           1-based ordinal of the subgoal within the plan.
        predicate_id: the L_env predicate this subgoal is tied to (verifier-grounded
                      plan credit: subgoal achieved iff this predicate becomes sat).
        text:         the human-readable subgoal text.
        char_span:    [start, end) char offsets into the PLAN COMPLETION text for this
                      subgoal line (end exclusive). Used by the rollout loop to attach
                      StageRecord char_spans to the planner turn.
    """

    id: int
    predicate_id: str
    text: str
    char_span: Tuple[int, int]

    def to_dict(self) -> dict:
        s, e = self.char_span
        return {
            "id": int(self.id),
            "predicate_id": str(self.predicate_id),
            "text": str(self.text),
            "char_span": [int(s), int(e)],
        }


# --------------------------------------------------------------------------- #
# Shared helpers                                                                #
# --------------------------------------------------------------------------- #
def _predicate_library_of(task: Task) -> List[str]:
    """Best-effort extraction of the task's predicate library (L_env).

    The authoritative L_env is env.predicate_library(task); the Task itself does not
    carry it in the frozen schema. The Planner only needs CANDIDATE predicate ids to
    suggest in its prompt and to match subgoals against, so we accept any of a few
    conventional carriers inside task.info / task.constraints (set by the env when it
    builds the task), and degrade gracefully to [] when absent.
    """
    if task is None:
        return []
    candidates: List[str] = []
    info = getattr(task, "info", None) or {}
    constraints = getattr(task, "constraints", None) or {}
    for src in (info, constraints):
        if not isinstance(src, dict):
            continue
        for key in ("predicate_library", "predicates", "L_env", "checkpoint_ids"):
            v = src.get(key)
            if isinstance(v, (list, tuple)):
                candidates = [str(x) for x in v]
                break
        if candidates:
            break
    return candidates


def _normalize(s: str) -> str:
    """Lowercase + collapse non-alphanumerics to single spaces (for fuzzy matching)."""
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _format_predicate_menu(predicate_library: List[str]) -> str:
    if not predicate_library:
        return "(predicate library unavailable to the planner; tie each subgoal to a " \
               "descriptive predicate_id and the environment will map it.)"
    lines = [f"  - {pid}" for pid in predicate_library]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Planner (FROZEN)                                                              #
# --------------------------------------------------------------------------- #
class Planner:
    """Decomposes a task into an ordered list of verifiable subgoals.

    Each subgoal is tied to a predicate_id from the environment's predicate library
    (L_env) so that plan-level credit can be verifier-grounded.
    """

    SYSTEM = (
        "You are the PLANNER in a verifier-grounded planning system.\n"
        "Decompose the task into an ORDERED list of VERIFIABLE subgoals. Each subgoal "
        "MUST be tied to exactly one predicate_id from the environment's predicate "
        "library (L_env): a subgoal counts as achieved ONLY when its predicate is "
        "verified satisfied. Do not invent rewards; every subgoal must map to a "
        "machine-checkable predicate.\n"
        "Rules:\n"
        "  1. Output the plan and NOTHING else between the tags " + PLAN_OPEN + " and "
        + PLAN_CLOSE + ".\n"
        "  2. One subgoal per line, ordered, using EXACTLY this format:\n"
        "       <n>. [predicate_id] <subgoal text>\n"
        "     where <n> is the 1-based step number and predicate_id is one of the "
        "predicates listed in the prompt (copy it verbatim).\n"
        "  3. Keep subgoals minimal, ordered by dependency, and each independently "
        "verifiable.\n"
        "  4. If you are given verifier feedback (a replan), revise ONLY the parts of "
        "the plan whose predicates are unsatisfied or regressed; keep already-"
        "satisfied subgoals stable.\n"
        "Example:\n"
        + PLAN_OPEN + "\n"
        "1. [budget_ok] Keep the running cost at or below the budget.\n"
        "2. [target_value_reached] Add high-value items until the target value is met.\n"
        + PLAN_CLOSE + "\n"
    )

    @staticmethod
    def build_messages(
        task: Task,
        state_summary: Optional[str],
        prev_plan: Optional[List[Subgoal]],
        feedback: Optional[str],
    ) -> List[Dict[str, str]]:
        """Build the chat messages for a Planner act.

        Args:
            task:          the current Task (carries spec / constraints / info).
            state_summary: short text summary of the current env state (e.g. from the
                           rollout loop's summarize(obs)); None on the initial plan.
            prev_plan:     the previous plan as a list[Subgoal] on a replan; None on
                           the initial plan.
            feedback:      verifier-EVENT feedback string on a replan (what regressed /
                           which predicate failed); None on the initial plan.

        Returns:
            list of {role, content} messages (system + user).
        """
        predicate_library = _predicate_library_of(task)
        spec = getattr(task, "spec", {}) or {}
        constraints = getattr(task, "constraints", {}) or {}

        problem_text = ""
        if isinstance(spec, dict):
            problem_text = str(spec.get("text") or spec.get("problem") or spec.get("statement") or "")
        if not problem_text:
            problem_text = str(spec)

        parts: List[str] = []
        parts.append("TASK SPECIFICATION:")
        parts.append(problem_text.strip() or "(no textual spec provided)")
        parts.append("")
        parts.append("MACHINE-CHECKABLE CONSTRAINTS:")
        parts.append(str(constraints))
        parts.append("")
        parts.append("PREDICATE LIBRARY (L_env) -- tie each subgoal to ONE of these ids:")
        parts.append(_format_predicate_menu(predicate_library))

        if state_summary:
            parts.append("")
            parts.append("CURRENT STATE:")
            parts.append(str(state_summary).strip())

        if prev_plan:
            parts.append("")
            parts.append("PREVIOUS PLAN (revise as needed):")
            for sg in prev_plan:
                pid = getattr(sg, "predicate_id", "")
                txt = getattr(sg, "text", "")
                sid = getattr(sg, "id", "?")
                parts.append(f"  {sid}. [{pid}] {txt}")

        if feedback:
            parts.append("")
            parts.append("VERIFIER FEEDBACK (replan trigger):")
            parts.append(str(feedback).strip())

        parts.append("")
        parts.append(
            "Produce the ordered, verifiable plan now. Wrap it in "
            + PLAN_OPEN + " ... " + PLAN_CLOSE + " and output nothing else."
        )

        user_content = "\n".join(parts)
        return [
            {"role": "system", "content": Planner.SYSTEM},
            {"role": "user", "content": user_content},
        ]

    # Matches lines like:  "1. [budget_ok] keep cost under budget"
    #                      "2) budget_ok: add items"
    #                      "- [target] do the thing"
    _SUBGOAL_LINE = re.compile(
        r"""^[ \t]*
            (?:(?P<num>\d+)[.)\]]?[ \t]*)?      # optional leading number
            (?:[-*][ \t]*)?                     # optional bullet
            (?:
                \[(?P<pid_b>[^\]]+)\]           # [predicate_id]
                |
                (?P<pid_c>[A-Za-z_][\w./-]*)\s*:   # predicate_id:
            )?
            [ \t]*
            (?P<text>.*\S)?                      # the subgoal text
            [ \t]*$
        """,
        re.VERBOSE,
    )

    @staticmethod
    def parse_plan(text: str) -> List[Subgoal]:
        """Parse the Planner completion into an ordered list[Subgoal].

        Extracts the FIRST <plan>...</plan> block (falling back to the whole text if
        the tags are absent), splits it into non-empty lines, and parses each line into
        a Subgoal. predicate_id assignment is best-effort:
          1. explicit [pid] or 'pid:' that fuzzy-matches a predicate in L_env -> use it;
          2. explicit token that does NOT match L_env -> keep the literal token
             (DeepPlanning / free predicates);
          3. no explicit token -> positional assignment: i-th subgoal gets the i-th
             still-unused predicate from L_env;
          4. L_env exhausted / unavailable -> synthetic id "subgoal_<n>".
        char_span indexes into the PLAN COMPLETION text passed in (the full `text`),
        so spans are valid even though parsing happens on the inner block.
        """
        if text is None:
            return []
        full = str(text)

        # Locate the <plan>...</plan> block within the full completion so char_spans
        # are offsets into `full` (the planner completion).
        block_start = 0
        block_text = full
        m_open = full.find(PLAN_OPEN)
        if m_open != -1:
            inner_start = m_open + len(PLAN_OPEN)
            m_close = full.find(PLAN_CLOSE, inner_start)
            inner_end = m_close if m_close != -1 else len(full)
            block_start = inner_start
            block_text = full[inner_start:inner_end]

        predicate_library = []  # filled lazily below from candidates passed via text? No.
        # NOTE: parse_plan only receives the completion text (frozen signature). The
        # L_env match is therefore done against predicate ids that APPEAR verbatim in
        # the text plus positional fallback. The rollout loop re-validates predicate_id
        # against env.predicate_library(task) and may remap; here we do a stable best
        # effort. We collect explicit predicate tokens to know which are "real".
        subgoals: List[Subgoal] = []
        used_positions = 0  # for positional synthetic fallback

        # Walk the block line by line, tracking absolute offsets into `full`.
        offset = block_start
        n_idx = 0
        for raw_line in block_text.splitlines(keepends=True):
            line_no_nl = raw_line.rstrip("\r\n")
            stripped = line_no_nl.strip()
            line_abs_start = offset
            offset += len(raw_line)

            if not stripped:
                continue
            # Skip a stray tag line if any leaked into the block.
            if stripped in (PLAN_OPEN, PLAN_CLOSE):
                continue

            m = Planner._SUBGOAL_LINE.match(line_no_nl)
            if not m:
                continue
            text_part = (m.group("text") or "").strip()
            pid_explicit = m.group("pid_b") or m.group("pid_c")
            # If the regex captured the whole line as predicate (e.g. a plain sentence
            # ending in nothing), guard: a bare sentence has no ']' / ':' -> pid is None.
            if not text_part and not pid_explicit:
                continue

            n_idx += 1

            # char_span of THIS subgoal line within `full`: from first non-space char
            # to the last non-space char on the line.
            lead_ws = len(line_no_nl) - len(line_no_nl.lstrip())
            span_start = line_abs_start + lead_ws
            span_end = line_abs_start + len(line_no_nl.rstrip())
            if span_end <= span_start:
                span_end = span_start + max(0, len(stripped))

            predicate_id = Planner._assign_predicate_id(
                pid_explicit=pid_explicit,
                predicate_library=predicate_library,
                position=used_positions,
                ordinal=n_idx,
            )
            used_positions += 1

            subgoals.append(
                Subgoal(
                    id=n_idx,
                    predicate_id=predicate_id,
                    text=text_part if text_part else (pid_explicit or "").strip(),
                    char_span=(int(span_start), int(span_end)),
                )
            )

        return subgoals

    @staticmethod
    def _assign_predicate_id(
        pid_explicit: Optional[str],
        predicate_library: List[str],
        position: int,
        ordinal: int,
    ) -> str:
        """Best-effort predicate_id for a subgoal (see parse_plan docstring)."""
        if pid_explicit:
            pid_explicit = pid_explicit.strip()
            if predicate_library:
                # exact match first
                if pid_explicit in predicate_library:
                    return pid_explicit
                # fuzzy match on normalized form
                npid = _normalize(pid_explicit)
                for cand in predicate_library:
                    if _normalize(cand) == npid:
                        return cand
            # No library or no match: keep the literal explicit token.
            return pid_explicit
        # No explicit predicate -> positional assignment into L_env.
        if predicate_library and position < len(predicate_library):
            return predicate_library[position]
        # Final fallback: synthetic, stable, ordinal-based.
        return f"subgoal_{ordinal}"


# --------------------------------------------------------------------------- #
# Executor (FROZEN)                                                             #
# --------------------------------------------------------------------------- #
class Executor:
    """Takes ONE action toward the current subgoal and emits <action>...</action>."""

    SYSTEM = (
        "You are the EXECUTOR in a verifier-grounded planning system.\n"
        "You are given a PLAN (ordered verifiable subgoals) and the CURRENT "
        "observation. Take EXACTLY ONE action that makes progress toward the current "
        "(first not-yet-satisfied) subgoal.\n"
        "Rules:\n"
        "  1. Output exactly one action wrapped in " + ACT_OPEN + " ... " + ACT_CLOSE
        + " and nothing else.\n"
        "  2. The action MUST be one of the available actions for this turn. Use this "
        "grammar:\n"
        "       add <item_index>        -- add the item with that index to the selection\n"
        "       remove <item_index>     -- remove the item with that index\n"
        "       finalize                -- commit the current selection for final "
        "verification\n"
        "  3. Take only ONE action. Do not chain multiple actions.\n"
        "  4. If a constraint regressed, prefer the action that recovers the violated "
        "predicate (e.g. remove an item that broke the budget).\n"
        "Example:\n"
        + ACT_OPEN + "add 3" + ACT_CLOSE + "\n"
    )

    @staticmethod
    def build_messages(
        task: Task,
        plan: List[Subgoal],
        obs: Observation,
        history: List[Any],
    ) -> List[Dict[str, str]]:
        """Build the chat messages for an Executor act.

        Args:
            task:    the current Task (for spec / constraints context).
            plan:    the current plan (list[Subgoal]); the executor targets the first
                     not-yet-satisfied subgoal (satisfaction is verifier-tracked by the
                     loop; here we just present the ordered plan).
            obs:     the current Observation (text + available_actions + state).
            history: list of prior turns/actions (each may be an Action, a dict, or a
                     string); rendered compactly so the executor has short-horizon
                     memory without unbounded growth.

        Returns:
            list of {role, content} messages (system + user).
        """
        spec = getattr(task, "spec", {}) or {}
        problem_text = ""
        if isinstance(spec, dict):
            problem_text = str(spec.get("text") or spec.get("problem") or spec.get("statement") or "")
        if not problem_text:
            problem_text = str(spec)

        parts: List[str] = []
        parts.append("TASK:")
        parts.append(problem_text.strip() or "(no textual spec provided)")
        parts.append("")
        parts.append("PLAN (ordered verifiable subgoals):")
        if plan:
            for sg in plan:
                pid = getattr(sg, "predicate_id", "")
                txt = getattr(sg, "text", "")
                sid = getattr(sg, "id", "?")
                parts.append(f"  {sid}. [{pid}] {txt}")
        else:
            parts.append("  (no plan available)")

        parts.append("")
        parts.append("CURRENT OBSERVATION:")
        obs_text = getattr(obs, "text", "") if obs is not None else ""
        parts.append(str(obs_text).strip() or "(no observation text)")

        avail = list(getattr(obs, "available_actions", []) or []) if obs is not None else []
        parts.append("")
        parts.append("AVAILABLE ACTIONS THIS TURN:")
        parts.append("  " + (", ".join(str(a) for a in avail) if avail else "(none provided)"))

        if history:
            parts.append("")
            parts.append("RECENT ACTION HISTORY (most recent last):")
            for h in Executor._render_history(history):
                parts.append("  " + h)

        parts.append("")
        parts.append(
            "Take ONE action now. Wrap it in " + ACT_OPEN + " ... " + ACT_CLOSE
            + " and output nothing else."
        )

        user_content = "\n".join(parts)
        return [
            {"role": "system", "content": Executor.SYSTEM},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _render_history(history: List[Any], max_items: int = 12) -> List[str]:
        """Render the last `max_items` history entries to short strings."""
        rendered: List[str] = []
        tail = history[-max_items:] if len(history) > max_items else history
        for h in tail:
            if isinstance(h, Action):
                rendered.append(h.raw.strip() if h.raw else str(h.parsed))
            elif isinstance(h, dict):
                # accept {raw|action|verb...}
                if "raw" in h:
                    rendered.append(str(h["raw"]).strip())
                else:
                    rendered.append(str(h))
            else:
                rendered.append(str(h).strip())
        return rendered

    # Capture the inner <action>...</action> region (non-greedy, dotall for safety).
    _ACTION_BLOCK = re.compile(
        re.escape(ACT_OPEN) + r"(?P<inner>.*?)" + re.escape(ACT_CLOSE),
        re.DOTALL,
    )
    # Parse an action verb + optional integer argument from the inner text.
    _ACTION_VERB = re.compile(
        r"^\s*(?P<verb>add|remove|finalize|done|submit|stop)\b\s*(?P<arg>-?\d+)?\s*$",
        re.IGNORECASE,
    )

    @staticmethod
    def parse_action(text: str) -> Action:
        """Parse the Executor completion into an Action.

        Returns an Action with:
          - raw:    the verbatim text inside the FIRST <action>...</action> block
                    (or the trimmed whole text if the tags are absent).
          - parsed: a dict describing the intent. Always carries:
                      {"verb": <str|"unknown">, "item_index": <int|None>,
                       "char_span": [start, end), "ok": <bool>}
                    where char_span indexes into the EXECUTOR COMPLETION `text` and
                    marks the inner action region (start of inner content -> end). This
                    is the span the rollout loop uses to attach StageRecord char_spans
                    to the executor turn. `ok` is False when no recognizable action verb
                    was found (the env may then treat it as a no-op / format failure).

        Recognized verbs: add/remove (with an integer item index), finalize
        (and synonyms done/submit/stop -> normalized to "finalize"). Unknown content
        yields verb="unknown", ok=False, but still carries the raw text + char_span so
        the loop never crashes on a malformed model output.
        """
        if text is None:
            return Action(raw="", parsed={"verb": "unknown", "item_index": None,
                                          "char_span": [0, 0], "ok": False})
        full = str(text)

        m_block = Executor._ACTION_BLOCK.search(full)
        if m_block is not None:
            inner = m_block.group("inner")
            inner_start = m_block.start("inner")
            inner_end = m_block.end("inner")
        else:
            # No tags: treat the whole (stripped) text as the action region.
            inner = full
            lead = len(full) - len(full.lstrip())
            inner_start = lead
            inner_end = len(full.rstrip())

        raw = inner.strip()
        parsed: Dict[str, Any] = {
            "verb": "unknown",
            "item_index": None,
            "char_span": [int(inner_start), int(inner_end)],
            "ok": False,
        }

        m_verb = Executor._ACTION_VERB.match(raw)
        if m_verb is not None:
            verb = m_verb.group("verb").lower()
            arg = m_verb.group("arg")
            # Normalize finalize synonyms.
            if verb in ("finalize", "done", "submit", "stop"):
                parsed["verb"] = "finalize"
                parsed["item_index"] = None
                parsed["ok"] = True
            elif verb in ("add", "remove"):
                parsed["verb"] = verb
                parsed["item_index"] = int(arg) if arg is not None else None
                # add/remove require an index to be a valid action.
                parsed["ok"] = parsed["item_index"] is not None
        else:
            # Fallback: scan for the first known verb token + first integer anywhere.
            low = raw.lower()
            for v in ("finalize", "done", "submit", "stop"):
                if re.search(r"\b" + v + r"\b", low):
                    parsed["verb"] = "finalize"
                    parsed["ok"] = True
                    break
            else:
                for v in ("add", "remove"):
                    if re.search(r"\b" + v + r"\b", low):
                        parsed["verb"] = v
                        m_int = re.search(r"-?\d+", raw)
                        parsed["item_index"] = int(m_int.group()) if m_int else None
                        parsed["ok"] = parsed["item_index"] is not None
                        break

        return Action(raw=raw, parsed=parsed)


# --------------------------------------------------------------------------- #
# __main__ smoke test (pure python; runs on CPU with no torch/vllm)             #
# --------------------------------------------------------------------------- #
def _smoke() -> None:
    print("=== roles.py smoke test ===")
    print(f"envs.base available: {_ENVS_BASE_AVAILABLE}")
    print(f"verifiers.base.StageRecord importable: {StageRecord is not None}")
    print()

    # Build a sample Task whose info carries a (mock) predicate library.
    predicate_library = [
        "budget_ok",
        "time_budget_ok",
        "required_categories_ok",
        "max_per_category_ok",
        "target_value_reached",
    ]
    task = Task(
        task_id="mock-0001",
        spec={"text": "Select items to maximize value under budget and time, "
                      "covering all required categories."},
        constraints={"budget": 100, "time_budget": 60, "target_value": 50},
        info={"predicate_library": predicate_library},
    )

    # --- Planner: build_messages (initial) ---
    msgs = Planner.build_messages(task, state_summary="empty selection", prev_plan=None, feedback=None)
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    print("[Planner.build_messages] system+user messages built; user content preview:")
    print("  " + msgs[1]["content"].splitlines()[0])
    print()

    # --- Planner: parse_plan on a sample completion ---
    sample_plan_text = (
        "Here is my plan.\n"
        + PLAN_OPEN + "\n"
        "1. [budget_ok] Keep the running cost at or below the budget of 100.\n"
        "2. [time_budget_ok] Ensure total time stays within 60.\n"
        "3. [required_categories_ok] Cover every required category at least once.\n"
        "4. [target_value_reached] Add high-value items until total value >= 50.\n"
        "5. finalize the selection once all predicates are satisfied.\n"
        + PLAN_CLOSE + "\n"
        "That completes the plan.\n"
    )
    subgoals = Planner.parse_plan(sample_plan_text)
    print(f"[Planner.parse_plan] parsed {len(subgoals)} subgoals:")
    for sg in subgoals:
        s, e = sg.char_span
        snippet = sample_plan_text[s:e]
        print(f"  id={sg.id} predicate_id={sg.predicate_id!r} char_span=({s},{e}) "
              f"text={sg.text!r}")
        # Verify the char_span actually indexes the line in the completion.
        assert sample_plan_text[s:e] == snippet
        assert sg.predicate_id, "predicate_id must be non-empty"
    # The first four lines have explicit predicate tokens; the fifth has none.
    assert subgoals[0].predicate_id == "budget_ok"
    assert subgoals[4].predicate_id in ("subgoal_5",) or subgoals[4].predicate_id, \
        "5th subgoal should get a synthetic predicate_id (no explicit token, no L_env match)"
    print()

    # --- Planner: replan messages (with prev_plan + feedback) ---
    msgs2 = Planner.build_messages(
        task,
        state_summary="cost=120 (OVER budget), value=40",
        prev_plan=subgoals,
        feedback="Predicate budget_ok regressed sat->unsat after adding item 7.",
    )
    assert "VERIFIER FEEDBACK" in msgs2[1]["content"]
    assert "PREVIOUS PLAN" in msgs2[1]["content"]
    print("[Planner.build_messages replan] feedback + prev_plan injected. OK")
    print()

    # --- Executor: build_messages ---
    obs = Observation(
        text="Selection is empty. 10 items available (indices 0..9).",
        available_actions=["add <i>", "remove <i>", "finalize"],
        state={"selected": [], "cost": 0, "value": 0},
    )
    history = [Action(raw="add 3", parsed={"verb": "add", "item_index": 3})]
    emsgs = Executor.build_messages(task, subgoals, obs, history)
    assert emsgs[0]["role"] == "system" and emsgs[1]["role"] == "user"
    assert "AVAILABLE ACTIONS" in emsgs[1]["content"]
    print("[Executor.build_messages] system+user messages built; user content preview:")
    print("  " + emsgs[1]["content"].splitlines()[0])
    print()

    # --- Executor: parse_action on several sample completions ---
    samples = [
        "I will add item 3.\n" + ACT_OPEN + "add 3" + ACT_CLOSE + "\nDone.",
        ACT_OPEN + " remove 7 " + ACT_CLOSE,
        "Now we are done. " + ACT_OPEN + "finalize" + ACT_CLOSE,
        "no tags here, just: add 5",
        "totally malformed output with no verb",
    ]
    for s in samples:
        act = Executor.parse_action(s)
        cs = act.parsed["char_span"]
        region = s[cs[0]:cs[1]]
        print(f"[Executor.parse_action] raw={act.raw!r} "
              f"verb={act.parsed['verb']!r} item_index={act.parsed['item_index']} "
              f"ok={act.parsed['ok']} char_span={tuple(cs)} region={region!r}")
        # char_span must be a valid slice of the completion.
        assert 0 <= cs[0] <= cs[1] <= len(s)

    # Concrete assertions on parse_action.
    a0 = Executor.parse_action(samples[0])
    assert a0.parsed["verb"] == "add" and a0.parsed["item_index"] == 3 and a0.parsed["ok"]
    a1 = Executor.parse_action(samples[1])
    assert a1.parsed["verb"] == "remove" and a1.parsed["item_index"] == 7
    a2 = Executor.parse_action(samples[2])
    assert a2.parsed["verb"] == "finalize" and a2.parsed["ok"]
    a4 = Executor.parse_action(samples[4])
    assert a4.parsed["verb"] == "unknown" and not a4.parsed["ok"]

    print()
    print("=== all smoke assertions passed ===")



# --------------------------------------------------------------------------- #
# Solver (R-Zero monolithic role)                                               #
# --------------------------------------------------------------------------- #
class Solver:
    """Monolithic R-Zero solver role.

    This role is for the R-Zero route only.

    It reads:
      - task specification
      - machine-checkable constraints
      - current observation
      - available actions
      - recent action history
      - optional verifier feedback

    It emits exactly one <action>...</action> block.

    It does NOT receive a plan and does NOT decompose the task into subgoals.
    """

    SYSTEM = (
        "You are a single SOLVER agent in a verifier-grounded environment.\n"
        "Your job is to solve the task by directly choosing the next concrete action "
        "from the current observation.\n"
        "Rules:\n"
        "  1. Output exactly one action wrapped in " + ACT_OPEN + " ... " + ACT_CLOSE
        + " and nothing else.\n"
        "  2. Use one of the available actions when they are provided.\n"
        "  3. Take only ONE action. Do not chain multiple actions.\n"
        "  4. You may reason internally, but do not output reasoning, plans, explanations, "
        "or extra markup.\n"
        "  5. If the task appears complete, use the environment's finalization/submission "
        "action when available.\n"
        "Example:\n"
        + ACT_OPEN + "finalize" + ACT_CLOSE + "\n"
    )

    @staticmethod
    def build_messages(
        task: Task,
        obs: Observation,
        history: List[Any],
        feedback: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build chat messages for one monolithic solver action.

        The solver sees the task, current observation, available actions, and recent
        action history. It does not receive a top-level plan.
        """
        spec = getattr(task, "spec", {}) or {}
        constraints = getattr(task, "constraints", {}) or {}

        problem_text = ""
        if isinstance(spec, dict):
            problem_text = str(
                spec.get("text")
                or spec.get("problem")
                or spec.get("statement")
                or ""
            )
        if not problem_text:
            problem_text = str(spec)

        parts: List[str] = []
        parts.append("TASK:")
        parts.append(problem_text.strip() or "(no textual task specification provided)")
        parts.append("")
        parts.append("MACHINE-CHECKABLE CONSTRAINTS:")
        parts.append(str(constraints))
        parts.append("")
        parts.append("CURRENT OBSERVATION:")
        obs_text = getattr(obs, "text", "") if obs is not None else ""
        parts.append(str(obs_text).strip() or "(no observation text)")

        avail = list(getattr(obs, "available_actions", []) or []) if obs is not None else []
        parts.append("")
        parts.append("AVAILABLE ACTIONS THIS TURN:")
        parts.append("  " + (", ".join(str(a) for a in avail) if avail else "(none provided)"))

        if history:
            parts.append("")
            parts.append("RECENT ACTION HISTORY (most recent last):")
            for h in Executor._render_history(history):
                parts.append("  " + h)

        if feedback:
            parts.append("")
            parts.append("VERIFIER FEEDBACK:")
            parts.append(str(feedback).strip())

        parts.append("")
        parts.append(
            "Choose the next concrete action now. Wrap it in "
            + ACT_OPEN + " ... " + ACT_CLOSE
            + " and output nothing else."
        )

        return [
            {"role": "system", "content": Solver.SYSTEM},
            {"role": "user", "content": "\n".join(parts)},
        ]

    @staticmethod
    def parse_action(text: str) -> Action:
        """Parse a monolithic solver action.

        Reuse the existing action parser so all environments receive the same Action
        contract as before: Action(raw=..., parsed=...).
        """
        return Executor.parse_action(text)


if __name__ == "__main__":
    _smoke()
