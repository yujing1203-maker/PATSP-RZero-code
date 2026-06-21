# -*- coding: utf-8 -*-
"""
Agentic multi-turn rollout loop + Policy abstraction -- FROZEN.

Provides:
  - Policy (ABC)      : act(messages) -> str.
  - RandomPolicy      : env-aware scripted/random valid <plan>/<action> emitter for the
                        dry-run path. PURE PYTHON -- no torch, no vllm. Runs on a CPU box
                        with neither installed.
  - VLLMPolicy        : vLLM + per-role LoRARequest. LAZY-imports vllm ONLY inside
                        __init__/act so importing this module never pulls in vllm.
  - run_episode(...)  : the multi-turn loop. Initial plan (boundary 0), executor acts,
                        env.step -> stage_records, verifier-EVENT replan (boundary++ when
                        env.needs_replan). Records PlannerTurn / ExecutorTurn, each with
                        prompt, completion, and stage_records whose char_span is set INTO
                        that turn's completion. terminal = env.verify_final().
  - Trajectory / PlannerTurn / ExecutorTurn dataclasses (-> JSONL rows).

The turns this loop emits are the SAME shape that split_agentic_rollouts_by_role.py turns
into the EXISTING per-row schema consumed UNCHANGED by
build_credit_targets.py / train_credit_head.py / apply_credit_head.py / the trainer.

Robust import: sys.path is fixed so `from verifiers.base import ...`,
`from envs.base import ...` and `from roles import ...` resolve regardless of cwd.

(The former mock-env CPU dry-run __main__ demo was retired with the mock env, 2026-06-11.)
"""

from __future__ import annotations

import abc
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --- Robust import ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verifiers.base import StageRecord  # noqa: E402,F401
from envs.base import Task, Observation, Action, StepResult, AgenticEnv  # noqa: E402
from roles import (  # noqa: E402
    Planner,
    Executor,
    Solver,
    Subgoal,
    PLAN_OPEN,
    PLAN_CLOSE,
    ACT_OPEN,
    ACT_CLOSE,
    _normalize,  # normalized-fuzzy matcher for predicate remapping (P4)
)

# NOTE: torch / vllm are intentionally NOT imported at module top. VLLMPolicy
# lazy-imports vllm inside __init__/act ONLY, so the dry-run path (RandomPolicy +
# MockConstraintPlanningEnv) runs on a CPU box with neither torch nor vllm installed.


# --------------------------------------------------------------------------- #
# Turn + Trajectory dataclasses                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class Turn:
    """One model turn (planner or executor). -> one JSONL row in agentic_trajectories.

    role:          "planner" | "executor".
    boundary:      replan boundary index (0..N). Increments on each verifier-event replan.
    turn_index:    monotone turn counter within the episode.
    prompt:        the rendered chat prompt (messages serialized) given to the policy.
    completion:    the model text returned by the policy (the <plan> or <action> output).
    stage_records: list[StageRecord]; char_span indexes INTO `completion`:
                     - planner: one record per subgoal predicate, char_span over the plan
                       region of the planner completion.
                     - executor: predicates that CHANGED this step, char_span over the
                       parsed action region of the executor completion.
    governed_turn_indices: PLANNER turns ONLY (P4). The list of EXECUTOR turn_index
                     values governed by this plan: from this plan boundary until the
                     next replan (a new PlannerTurn) or the terminal step. Empty for
                     executor turns. The split (P3) reads this to compute the planner
                     credit A_P as the group-relative SUM of executor-achieved stage
                     potentials Phi_i over exactly these governed turns. Additive /
                     backward-compatible: defaults to [] so old readers ignore it.
    info:          free-form (action parse, replan feedback, step info, ...).
    """

    role: str
    boundary: int
    turn_index: int
    prompt: str
    completion: str
    stage_records: List[StageRecord] = field(default_factory=list)
    governed_turn_indices: List[int] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "boundary": int(self.boundary),
            "turn_index": int(self.turn_index),
            "prompt": self.prompt,
            "completion": self.completion,
            "stage_records": [r.to_dict() for r in self.stage_records],
            "governed_turn_indices": [int(i) for i in self.governed_turn_indices],
            "info": dict(self.info),
        }


# PlannerTurn / ExecutorTurn are role-tagged Turns. Keeping them as thin subclasses
# lets callers `isinstance(turn, PlannerTurn)` while the JSONL row stays a single
# uniform schema (role field distinguishes them).
class PlannerTurn(Turn):
    def __init__(self, **kw):
        kw["role"] = "planner"
        super().__init__(**kw)


class ExecutorTurn(Turn):
    def __init__(self, **kw):
        kw["role"] = "executor"
        super().__init__(**kw)


class SolverTurn(Turn):
    def __init__(self, **kw):
        kw["role"] = "solver"
        super().__init__(**kw)


@dataclass
class Trajectory:
    """One agentic rollout.

    turns:                 ordered list[Turn] (planner + executor).
    terminal_reward:       env.verify_final() in [0,1].
    predicate_library:     L_env for this task (for the JSONL row + downstream credit).
    stage_records_global:  flat list of all stage_records across executor turns (the
                           verifier event stream for the episode), char_spans LOCAL to
                           their own turn's completion (do NOT re-index globally).
    info:                  free-form (n_replans, max_steps reached, ...).
    """

    task_id: str
    turns: List[Turn]
    terminal_reward: float
    predicate_library: List[str]
    stage_records_global: List[StageRecord] = field(default_factory=list)
    sample_index: int = 0
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "sample_index": int(self.sample_index),
            "terminal_reward": float(self.terminal_reward),
            "turns": [t.to_dict() for t in self.turns],
            "predicate_library": list(self.predicate_library),
            "stage_records_global": [r.to_dict() for r in self.stage_records_global],
            "info": dict(self.info),
        }


# --------------------------------------------------------------------------- #
# Policy abstraction                                                            #
# --------------------------------------------------------------------------- #
class Policy(abc.ABC):
    """A text policy: maps a chat `messages` list to a model-text completion."""

    @abc.abstractmethod
    def act(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Serialize chat messages into a single prompt string (stored on the Turn).

    Simple, deterministic role-tagged concatenation; the real chat template is applied
    inside VLLMPolicy. Used for the stored `prompt` field and as a fallback prompt.
    """
    parts = []
    for m in messages:
        parts.append(f"<|{m.get('role', 'user')}|>\n{m.get('content', '')}")
    return "\n".join(parts)


class RandomPolicy(Policy):
    """Env-aware scripted/random policy for the dry-run path (CPU, NO torch/vllm).

    It reads the env's current state (selection / item indices / predicate library) so it
    emits VALID <plan>...</plan> and <action>...</action> text that roles.parse_plan /
    roles.parse_action accept. The plan ties subgoals to real L_env predicate ids; the
    actions walk a feasibility-seeking heuristic (add value-dense feasible items, remove
    a violating item on regress, finalize when all hard predicates are sat) with some
    randomness so multiple samples per task differ (group-relative advantage).
    """

    def __init__(self, env: AgenticEnv, task: Optional[Task] = None, seed: int = 0):
        self.env = env
        self.task = task
        self.rng = random.Random(seed)

    def bind_task(self, task: Task) -> None:
        self.task = task

    def act(self, messages: List[Dict[str, str]]) -> str:
        """Dispatch on the system prompt: planner -> emit a <plan>, executor -> <action>."""
        system = messages[0].get("content", "") if messages else ""
        is_planner = "PLANNER" in system or "Planner" in system or "decompose" in system.lower()
        if is_planner:
            return self._emit_plan()
        return self._emit_action()

    # ---- planner output ----
    def _emit_plan(self) -> str:
        pids: List[str] = []
        if self.task is not None:
            try:
                pids = list(self.env.predicate_library(self.task))
            except Exception:
                pids = []
        lines = [PLAN_OPEN]
        if pids:
            for i, pid in enumerate(pids, start=1):
                lines.append(f"{i}. [{pid}] Make predicate {pid} satisfied.")
        else:
            lines.append("1. [subgoal_1] Satisfy the task constraints.")
        # last subgoal: finalize once predicates hold
        lines.append(f"{len(pids) + 1 if pids else 2}. finalize once all predicates are satisfied.")
        lines.append(PLAN_CLOSE)
        return "\n".join(lines)

    # ---- executor output ----
    def _emit_action(self) -> str:
        """Pick a valid action from the live env state and wrap it in <action> tags."""
        state = self._env_state()
        items = self._items()
        selection = state.get("selection", [])
        n_items = state.get("n_items", len(items))

        action_str = self._choose_action(items, selection, n_items)
        return f"{ACT_OPEN}{action_str}{ACT_CLOSE}"

    def _choose_action(self, items, selection, n_items) -> str:
        # If env says a predicate regressed, prefer removing a likely culprit.
        # (RandomPolicy is a heuristic, not optimal -- that's fine for dry-run plumbing.)
        c = self.task.constraints if self.task is not None else {}
        # Compute current cost/time to decide add vs remove vs finalize.
        cost = sum(int(items[i]["cost"]) for i in selection if 0 <= i < len(items))
        time = sum(int(items[i]["time"]) for i in selection if 0 <= i < len(items))
        budget = int(c.get("budget", 10 ** 9))
        time_budget = int(c.get("time_budget", 10 ** 9))

        over = cost > budget or time > time_budget

        # Finalize sometimes once a non-empty, in-budget selection exists.
        if selection and not over and self.rng.random() < 0.35:
            return "finalize"

        if over and selection:
            # remove a random selected item to recover budget/time.
            victim = self.rng.choice(sorted(set(selection)))
            return f"remove {victim}"

        # else add a not-yet-selected feasible-ish item, preferring value density.
        candidates = [i for i in range(n_items) if i not in selection]
        if not candidates:
            return "finalize"
        # value density ordering with a little randomness
        self.rng.shuffle(candidates)
        candidates.sort(
            key=lambda i: -(int(items[i]["value"]) / max(1, int(items[i]["cost"]) + int(items[i]["time"])))
            if 0 <= i < len(items) else 0.0
        )
        pick = candidates[0]
        return f"add {pick}"

    # ---- env introspection helpers ----
    def _env_state(self) -> dict:
        try:
            obs = self.env._make_observation()  # mock env exposes current observation
            return obs.state
        except Exception:
            return {}

    def _items(self) -> List[dict]:
        if self.task is not None:
            return self.task.spec.get("items", [])
        return []


class VLLMPolicy(Policy):
    """vLLM-backed policy with a per-role LoRA adapter.

    LAZY-imports vllm ONLY inside __init__/act so importing this module is torch/vllm-free.
    role in {"planner","executor","solver","challenger"} selects the LoRA int_id (planner=41, executor=42, solver=43, challenger=44).
    """

    ROLE_INT_IDS = {"planner": 41, "executor": 42, "solver": 43, "challenger": 44}

    def __init__(
        self,
        role: str,
        lora: Optional[str] = None,
        model: Optional[str] = None,
        llm: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        gpu_memory_utilization: float = 0.85,
        enable_lora: bool = True,
    ):
        if role not in self.ROLE_INT_IDS:
            raise ValueError(f"role must be one of {list(self.ROLE_INT_IDS)}, got {role!r}")
        self.role = role
        self.lora_path = lora
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        # Lazy import: keep module import torch/vllm-free for the dry-run path.
        from vllm import LLM, SamplingParams  # noqa: F401  (vllm only here)
        from vllm.lora.request import LoRARequest

        self._SamplingParams = SamplingParams
        self._LoRARequest = LoRARequest

        if llm is not None:
            self.llm = llm
        else:
            if model is None:
                raise ValueError("VLLMPolicy needs either an existing `llm` or a `model` path.")
            # Shared-GPU stability knobs (env-overridable; infra only, no effect on
            # rollout semantics). Lightweight engine: cap context, small warmup batch,
            # skip CUDA-graph capture. PLUS retry the engine init -- on a shared GPU,
            # bursty neighbor jobs cause transient vLLM memory-profiling failures
            # ("No available memory for the cache blocks" / profiling assertion /
            # warmup OOM); retrying after a short delay lands in a quieter window.
            import os as _os, time as _time, sys as _sys, gc as _gc
            _retries = max(1, int(_os.environ.get("RZERO_VLLM_INIT_RETRIES", "6")))
            _delay = float(_os.environ.get("RZERO_VLLM_INIT_RETRY_DELAY", "25"))
            _err = None
            for _i in range(_retries):
                try:
                    self.llm = LLM(
                        model=model,
                        enable_lora=enable_lora,  # full-param engines load a whole checkpoint; no adapter
                        tensor_parallel_size=int(_os.environ.get("RZERO_VLLM_TP", "1")),  # dual-card TP via RZERO_VLLM_TP
                        trust_remote_code=True,
                        gpu_memory_utilization=gpu_memory_utilization,
                        max_model_len=int(_os.environ.get("RZERO_STAGE_MAX_MODEL_LEN", "2048")),
                        max_num_seqs=int(_os.environ.get("RZERO_VLLM_MAX_NUM_SEQS", "16")),
                        enforce_eager=_os.environ.get("RZERO_VLLM_ENFORCE_EAGER", "1") == "1",
                    )
                    _err = None
                    break
                except Exception as _e:  # transient shared-GPU contention at init
                    _err = _e
                    print(
                        f"[VLLMPolicy] vLLM engine init attempt {_i + 1}/{_retries} failed "
                        f"({type(_e).__name__}: {_e}); shared-GPU contention, retrying in {_delay}s.",
                        file=_sys.stderr, flush=True,
                    )
                    _gc.collect()
                    try:
                        import torch as _torch
                        _torch.cuda.empty_cache()
                    except Exception:
                        pass
                    _time.sleep(_delay)
            if _err is not None:
                raise _err

        # Tokenizer for chat templating; reuse the LLM's if not supplied.
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            try:
                self.tokenizer = self.llm.get_tokenizer()
            except Exception:
                self.tokenizer = None

        self._lora_request = None
        if self.lora_path:
            self._lora_request = self._LoRARequest(
                f"{self.role}_adapter",
                self.ROLE_INT_IDS[self.role],
                self.lora_path,
            )

    def _render_prompt(self, messages: List[Dict[str, str]]) -> str:
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return _messages_to_prompt(messages)

    def act(self, messages: List[Dict[str, str]]) -> str:
        prompt = self._render_prompt(messages)
        # Optional repetition_penalty guards against the GRPO round-2 degeneration where the
        # executor collapses to long repeated-digit item indices (e.g. `add 70030000...0`),
        # which go out-of-range and zero the reward. Off by default (1.0); set RZERO_REPETITION_PENALTY.
        _sp_kw = dict(max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p)
        _rp = float(os.environ.get("RZERO_REPETITION_PENALTY", "1.0"))
        if _rp and _rp != 1.0:
            _sp_kw["repetition_penalty"] = _rp
        sp = self._SamplingParams(**_sp_kw)
        kwargs = {}
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request
        outputs = self.llm.generate([prompt], sp, **kwargs)
        return outputs[0].outputs[0].text


# --------------------------------------------------------------------------- #
# stage-record char_span attachment helpers                                     #
# --------------------------------------------------------------------------- #
def _attach_span(records: List[StageRecord], span: Tuple[int, int]) -> List[StageRecord]:
    """Return copies of `records` with char_span set to `span` (INTO the turn completion)."""
    s, e = int(span[0]), int(span[1])
    out: List[StageRecord] = []
    for r in records:
        out.append(StageRecord(
            predicate_id=r.predicate_id,
            before=r.before,
            after=r.after,
            failure_type=r.failure_type,
            char_span=(s, max(s, e)),
            info=dict(r.info),
        ))
    return out


def _remap_subgoals_to_library(
    subgoals: List[Subgoal],
    predicate_library: List[str],
) -> Tuple[List[Subgoal], List[Subgoal]]:
    """Remap each Subgoal.predicate_id to the nearest L_env predicate id (P4).

    For each subgoal, match its predicate_id against env.predicate_library(task):
      1. EXACT match -> keep the library id verbatim.
      2. else NORMALIZED-FUZZY match (roles._normalize collapses non-alphanumerics +
         lowercases) against each library id -> use the library id.
      3. no match -> the subgoal is DROPPED from the matched set and returned in the
         `unmatched` list (flagged), so it never enters the planner stage_records and
         therefore never grounds plan-level credit against a non-L_env predicate.

    This ties plan subgoals to real L_env predicates (the verifier vocabulary) for a real
    model whose free-text predicate tokens need not equal the library ids verbatim.

    Returns (matched_subgoals, unmatched_subgoals). matched_subgoals carry the remapped
    (canonical) predicate_id; order is preserved.
    """
    if not predicate_library:
        # No L_env to ground against: pass subgoals through unchanged (best effort).
        return list(subgoals), []

    norm_lib: Dict[str, str] = {}
    for pid in predicate_library:
        norm_lib.setdefault(_normalize(pid), pid)

    matched: List[Subgoal] = []
    unmatched: List[Subgoal] = []
    for sg in subgoals:
        raw_pid = (sg.predicate_id or "").strip()
        canonical: Optional[str] = None
        if raw_pid in predicate_library:
            canonical = raw_pid
        else:
            canonical = norm_lib.get(_normalize(raw_pid))
        if canonical is None:
            unmatched.append(sg)
            continue
        matched.append(Subgoal(
            id=sg.id,
            predicate_id=canonical,
            text=sg.text,
            char_span=(int(sg.char_span[0]), int(sg.char_span[1])),
        ))
    return matched, unmatched


def _subgoals_to_stage_records(
    subgoals: List[Subgoal],
    achieved_predicates: set,
) -> List[StageRecord]:
    """Planner stage_records = one record per subgoal predicate, char_span over the plan
    region (the subgoal's own line). after='sat' iff that predicate was ACTUALLY ACHIEVED
    by the executor segment this plan governed (verifier-grounded over the governed turns,
    NOT trivially-sat at plan time), else 'unsat'. failure_type='none' for sat, 'missing'
    for unsat (the plan asserted the subgoal but the governed segment never achieved it).

    NOTE: subgoals are expected to already be remapped to L_env ids via
    _remap_subgoals_to_library; `achieved_predicates` is the set of predicate_ids that the
    governed executor segment drove to 'sat' (computed in run_episode after the segment
    closes). This makes the split A_P (P3) non-vacuous: a plan whose governed segment
    achieved its subgoals scores higher than one whose segment did not."""
    records: List[StageRecord] = []
    for sg in subgoals:
        if not sg.predicate_id:
            continue
        after = "sat" if sg.predicate_id in achieved_predicates else "unsat"
        records.append(StageRecord(
            predicate_id=sg.predicate_id,
            before="unsat",
            after=after,
            failure_type="none" if after == "sat" else "missing",
            char_span=(int(sg.char_span[0]), int(sg.char_span[1])),
            info={"subgoal_id": sg.id, "subgoal_text": sg.text},
        ))
    return records


def _action_span(action: Action, completion: str) -> Tuple[int, int]:
    """Get the parsed action region char_span within `completion`.

    roles.Executor.parse_action stores it in action.parsed["char_span"]; clamp to the
    completion bounds for safety. Falls back to the whole completion if absent.
    """
    span = (action.parsed or {}).get("char_span")
    if isinstance(span, (list, tuple)) and len(span) == 2:
        s = max(0, min(int(span[0]), len(completion)))
        e = max(s, min(int(span[1]), len(completion)))
        return (s, e)
    return (0, len(completion))


def summarize(obs: Observation) -> str:
    """Short text state summary handed to the Planner on (re)plan."""
    if obs is None:
        return "(start)"
    st = obs.state or {}
    totals = st.get("totals", {})
    return (
        f"selection={st.get('selection')} totals={totals} "
        f"finalized={st.get('finalized')}"
    )


def _replan_feedback(last: StepResult) -> str:
    """Verifier-event feedback string for the Planner on a replan."""
    if last is None:
        return ""
    regressed = []
    for r in last.stage_records:
        if r.before == "sat" and r.after == "unsat":
            regressed.append(f"{r.predicate_id} (sat->unsat, {r.failure_type})")
    if regressed:
        return "Predicate(s) regressed this step: " + "; ".join(regressed) + \
               ". Revise the plan to recover them."
    if last.info.get("invalid_action"):
        return "Last action was invalid: " + str(last.info.get("invalid_reason", "")) + \
               ". Re-plan a valid next step."
    return "A verifier event requires re-planning."



# --------------------------------------------------------------------------- #
# The monolithic solver loop -- R-Zero route                                    #
# --------------------------------------------------------------------------- #
def _solver_feedback(last: Optional[StepResult]) -> Optional[str]:
    """Verifier feedback for the monolithic solver.

    This is not a re-planning instruction. It is only compact feedback about the
    last environment transition so the single solver can choose a recovery action.
    """
    if last is None:
        return None

    regressed = []
    for r in last.stage_records:
        if r.before == "sat" and r.after == "unsat":
            regressed.append(f"{r.predicate_id} changed sat->unsat ({r.failure_type})")

    if regressed:
        return "Verifier feedback: " + "; ".join(regressed) + ". Choose a recovery action."

    if last.info.get("invalid_action"):
        return "Verifier feedback: last action was invalid: " + str(
            last.info.get("invalid_reason", "")
        )

    if last.info:
        return "Verifier feedback: " + str(last.info)

    return "Verifier feedback: previous action did not complete the task."


def run_solver_episode(
    env: AgenticEnv,
    task: Task,
    solver: Any = Solver,
    solver_policy: Policy = None,
    max_steps: int = 16,
    sample_index: int = 0,
) -> Trajectory:
    """Run one monolithic R-Zero solver episode.

    Unlike run_episode(...), this function does not create a plan and does not
    call an executor. A single solver policy directly emits one environment
    action per step.

    Output trajectory turns have role="solver".
    """
    if solver_policy is None:
        raise ValueError("run_solver_episode requires solver_policy")

    predicate_library = list(env.predicate_library(task))
    task.info = dict(task.info)
    task.info.setdefault("predicate_library", predicate_library)

    if isinstance(solver_policy, RandomPolicy):
        solver_policy.bind_task(task)

    obs = env.reset(task)
    turns: List[Turn] = []
    history: List[dict] = []
    stage_records_global: List[StageRecord] = []

    turn_index = 0
    last: Optional[StepResult] = None
    done = False
    n_verifier_feedback_events = 0

    for _step in range(max_steps):
        feedback = None
        if last is not None and env.needs_replan(last):
            feedback = _solver_feedback(last)
            n_verifier_feedback_events += 1

        solver_msgs = solver.build_messages(
            task=task,
            obs=obs,
            history=history,
            feedback=feedback,
        )
        act_text = solver_policy.act(solver_msgs)
        action = solver.parse_action(act_text)

        step = env.step(action)

        span = _action_span(action, act_text)
        solver_records = _attach_span(step.stage_records, span)
        stage_records_global.extend(solver_records)

        turns.append(SolverTurn(
            boundary=0,
            turn_index=turn_index,
            prompt=_messages_to_prompt(solver_msgs),
            completion=act_text,
            stage_records=solver_records,
            info={
                "action_parsed": dict(action.parsed),
                "step_info": dict(step.info),
                "done": bool(step.done),
                "feedback": feedback,
            },
        ))

        history.append({
            "turn_index": turn_index,
            "action_raw": action.raw,
            "parsed": dict(action.parsed),
        })

        turn_index += 1
        obs = step.obs
        last = step

        if step.done:
            done = True
            break

    terminal = float(env.verify_final())

    return Trajectory(
        task_id=task.task_id,
        turns=turns,
        terminal_reward=terminal,
        predicate_library=predicate_library,
        stage_records_global=stage_records_global,
        sample_index=int(sample_index),
        info={
            "n_turns": len(turns),
            "mode": "monolithic_solver",
            "max_steps": max_steps,
            "reached_done": done,
            "n_verifier_feedback_events": n_verifier_feedback_events,
        },
    )


# --------------------------------------------------------------------------- #
# The multi-turn loop -- FROZEN signature                                       #
# --------------------------------------------------------------------------- #
def run_episode(
    env: AgenticEnv,
    task: Task,
    planner: Any,
    executor: Any,
    planner_policy: Policy,
    executor_policy: Policy,
    max_steps: int = 16,
    max_replans: int = 3,
    sample_index: int = 0,
) -> Trajectory:
    """Run one agentic episode and return a Trajectory.

    Loop:
      - Initial plan at boundary 0.
      - Each step: if env.needs_replan(last) and boundary < max_replans -> replan
        (boundary++), recording a PlannerTurn.
      - Executor acts -> ExecutorTurn; env.step -> stage_records (char_span set INTO the
        executor completion's parsed-action region).
      - terminal = env.verify_final() at the end.

    planner / executor are the role helpers (roles.Planner / roles.Executor); they expose
    build_messages(...) and parse_plan / parse_action. planner_policy / executor_policy are
    Policy instances (RandomPolicy on the dry-run path).
    """
    # Expose L_env to the Planner prompt builder via task.info (roles reads task.info).
    predicate_library = list(env.predicate_library(task))
    task.info = dict(task.info)
    task.info.setdefault("predicate_library", predicate_library)

    # Bind the live task to RandomPolicy instances so they can emit env-valid output.
    for pol in (planner_policy, executor_policy):
        if isinstance(pol, RandomPolicy):
            pol.bind_task(task)

    obs = env.reset(task)
    turns: List[Turn] = []
    history: List[dict] = []
    stage_records_global: List[StageRecord] = []
    turn_index = 0
    boundary = 0
    last: Optional[StepResult] = None

    # P4 bookkeeping. For each PlannerTurn we record the matched (L_env-grounded)
    # subgoals so its stage_records can be built AFTER its governed executor segment is
    # known (HIGH: planner record `after` = whether the segment ACTUALLY achieved the
    # predicate, not trivially-sat at plan time). achieved_at[turn_index] = set of
    # predicate_ids the executor drove to 'sat' at that executor turn.
    planner_turns: List[Tuple[PlannerTurn, List[Subgoal]]] = []
    achieved_at: Dict[int, set] = {}
    n_unmatched_subgoals = 0

    def _record_plan(
        prompt: str,
        plan_text: str,
        matched: List[Subgoal],
        n_parsed: int,
        unmatched: List[Subgoal],
        evt_info: dict,
    ) -> None:
        """Record a PlannerTurn whose subgoals are already L_env-grounded (P4). Its
        stage_records are back-filled later from its governed segment; stash the matched
        subgoals so the back-fill knows which predicates the plan asserted."""
        nonlocal n_unmatched_subgoals
        n_unmatched_subgoals += len(unmatched)
        info = dict(evt_info)
        info["n_subgoals"] = int(n_parsed)
        info["n_matched_subgoals"] = len(matched)
        info["n_unmatched_subgoals"] = len(unmatched)
        if unmatched:
            info["unmatched_predicate_ids"] = [sg.predicate_id for sg in unmatched]
        pt = PlannerTurn(
            boundary=boundary,
            turn_index=turn_index,
            prompt=prompt,
            completion=plan_text,
            stage_records=[],  # back-filled after the governed segment closes
            governed_turn_indices=[],  # back-filled after the loop
            info=info,
        )
        turns.append(pt)
        planner_turns.append((pt, matched))

    # --- initial plan (boundary 0) ---
    plan_msgs = planner.build_messages(task, summarize(obs), None, None)
    plan_text = planner_policy.act(plan_msgs)
    parsed_plan: List[Subgoal] = planner.parse_plan(plan_text)
    # Ground subgoals to L_env BEFORE they drive the executor / planner stage_records.
    plan, unmatched = _remap_subgoals_to_library(parsed_plan, predicate_library)
    _record_plan(
        _messages_to_prompt(plan_msgs), plan_text, plan, len(parsed_plan), unmatched,
        {"event": "initial_plan"},
    )
    turn_index += 1

    done = False
    for _step in range(max_steps):
        # --- verifier-EVENT replan ---
        if last is not None and env.needs_replan(last) and boundary < max_replans:
            boundary += 1
            feedback = _replan_feedback(last)
            replan_msgs = planner.build_messages(task, summarize(obs), plan, feedback)
            replan_text = planner_policy.act(replan_msgs)
            parsed_replan = planner.parse_plan(replan_text)
            plan, unmatched_r = _remap_subgoals_to_library(parsed_replan, predicate_library)
            _record_plan(
                _messages_to_prompt(replan_msgs), replan_text, plan, len(parsed_replan),
                unmatched_r, {"event": "replan", "feedback": feedback},
            )
            turn_index += 1

        # --- executor acts ---
        exec_msgs = executor.build_messages(task, plan, obs, history)
        act_text = executor_policy.act(exec_msgs)
        action = executor.parse_action(act_text)

        step = env.step(action)

        # Attach env stage_records' char_span INTO this executor completion's action region.
        span = _action_span(action, act_text)
        exec_records = _attach_span(step.stage_records, span)
        stage_records_global.extend(exec_records)

        # P4 grounding: predicates this executor turn drove to 'sat' (first attainment OR
        # recovery). Used to back-fill the governing plan's stage_record `after`.
        achieved_at[turn_index] = {
            r.predicate_id for r in exec_records if r.after == "sat"
        }

        turns.append(ExecutorTurn(
            boundary=boundary,
            turn_index=turn_index,
            prompt=_messages_to_prompt(exec_msgs),
            completion=act_text,
            stage_records=exec_records,
            info={
                "action_parsed": dict(action.parsed),
                "step_info": dict(step.info),
                "done": bool(step.done),
            },
        ))
        turn_index += 1

        history.append({
            "turn_index": turn_index - 1,
            "action_raw": action.raw,
            "parsed": dict(action.parsed),
        })
        obs = step.obs
        last = step

        if step.done:
            done = True
            break

    # --- P4: assign each plan its governed executor segment + ground its stage_records ---
    # A plan at planner turn_index p governs every EXECUTOR turn from p (exclusive) until
    # the NEXT planner turn (exclusive) or the terminal step. We then build the planner's
    # stage_records with after='sat' iff the governed segment actually achieved that
    # subgoal predicate (union of achieved_at over the governed executor turns).
    planner_turn_indices = sorted(pt.turn_index for pt, _ in planner_turns)
    exec_turn_indices = sorted(achieved_at.keys())
    for pt, matched in planner_turns:
        # boundary of governance: next planner turn_index after this one (or +inf).
        later = [ti for ti in planner_turn_indices if ti > pt.turn_index]
        next_planner_ti = later[0] if later else None
        governed = [
            ti for ti in exec_turn_indices
            if ti > pt.turn_index and (next_planner_ti is None or ti < next_planner_ti)
        ]
        achieved: set = set()
        for ti in governed:
            achieved |= achieved_at.get(ti, set())
        pt.governed_turn_indices = list(governed)
        pt.stage_records = _subgoals_to_stage_records(matched, achieved)
        pt.info["n_achieved_subgoals"] = sum(
            1 for r in pt.stage_records if r.after == "sat"
        )

    terminal = float(env.verify_final())

    return Trajectory(
        task_id=task.task_id,
        turns=turns,
        terminal_reward=terminal,
        predicate_library=predicate_library,
        stage_records_global=stage_records_global,
        sample_index=int(sample_index),
        info={
            "n_turns": len(turns),
            "n_replans": boundary,
            "max_steps": max_steps,
            "max_replans": max_replans,
            "reached_done": done,
            "n_unmatched_subgoals": n_unmatched_subgoals,
        },
    )


# --------------------------------------------------------------------------- #
# __main__ : CPU dry-run on MockConstraintPlanningEnv + RandomPolicy (no model) #
# --------------------------------------------------------------------------- #
def _main() -> None:
    # The old CPU dry-run demo (MockConstraintPlanningEnv + RandomPolicy) was retired together with
    # the mock env (2026-06-11 restructure; GPU-only policy). Use build_agentic_rollouts.py with
    # --env textcraft / scienceworld / alfworld / herobench instead.
    print("rollout_agentic: dry-run demo removed; use build_agentic_rollouts.py with a real env.")


if __name__ == "__main__":
    _main()
