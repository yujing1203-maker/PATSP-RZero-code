# -*- coding: utf-8 -*-
"""ScienceWorld adapter -> envs.base.AgenticEnv  (benchmark #2 for PATSP).

ScienceWorld (Wang et al. 2022) is a text simulator of elementary-science procedures
(boil a substance, grow a plant, measure conductivity, ...). It is an ideal PATSP substrate:

  * `get_goal_progress()` exposes the task's **sequential sub-goals** with per-sub-goal
    completion -> we map these directly to predicate_library + StageRecords. The ordering is a
    real DEPENDENCY (you must `focus on substance` before you can change its state), so
    subgoal-ordering / premature-stop / missing-prerequisite failures are first-class.
  * `score` in [0,100] gives continuous partial credit -> verify_final = score/100 (never
    trivially 1.0, real learning signal).
  * A sub-goal can regress (undone), so RECOVERY (re-satisfying it) is observable -- the
    trajectory-level signal PATSP trains on.

Install: `pip install --no-deps scienceworld py4j` (jar auto-downloads on first env; needs java).
Pure-import-safe: the heavy java env is created lazily on first reset().
"""

from __future__ import annotations

import os
import re
import random
from typing import Any, Dict, List, Optional, Tuple

from envs.base import Task, Observation, Action, StepResult, AgenticEnv
from verifiers.base import StageRecord

# A difficulty ladder over ScienceWorld task types, short-horizon -> long-horizon.
# (Names must match scienceworld get_task_names(); these are the canonical task ids.)
_LADDER = [
    "find-living-thing",                 # navigation only (easiest)
    "find-non-living-thing",
    "boil",                              # focus -> heat -> state change
    "melt",
    "freeze",
    "change-the-state-of-matter-of",
    "measure-melting-point-known-substance",
    "chemistry-mix",                     # gather -> combine
    "power-component",                   # build/connect a circuit
    "grow-plant",                        # long multi-stage (hardest)
]

_STEP_LIMIT = int(os.environ.get("SCIENCEWORLD_STEP_LIMIT", "40"))


def _parse_action(raw: str) -> str:
    """Extract the ScienceWorld command from the executor's text."""
    t = (raw or "").strip()
    m = re.search(r"<action>(.*?)</action>", t, flags=re.S)
    if m:
        t = m.group(1).strip()
    # take the first non-empty line; drop a leading 'action:' label if present
    line = next((ln.strip() for ln in t.splitlines() if ln.strip()), t)
    line = re.sub(r"^(action|command)\s*[:\-]\s*", "", line, flags=re.I).strip()
    return line[:200]


def _parse_goal_progress(s: str) -> List[Dict[str, Any]]:
    """Parse get_goal_progress() into per-line subgoals, enumerated in appearance order.

    ScienceWorld can list several alternative solution paths, each re-numbered from 0, so the
    file's own idx column collides. We assign a GLOBAL position id (pid=sg{i}) per parsed line
    -- one stable, unique predicate per sub-goal line (order is stable across steps).
    """
    out: List[Dict[str, Any]] = []
    for ln in (s or "").splitlines():
        m = re.match(r"\s*(\d+)\s+(true|false)\s+(\S+)\s+(.*\S)\s*$", ln)
        if m:
            i = len(out)
            out.append({"pid": f"sg{i}", "done": m.group(2) == "true",
                        "kind": m.group(3), "desc": m.group(4).strip()})
    return out


def _scienceworld_command_guidance(instr: str, sw_obs: str) -> str:
    """Task-specific command grounding hints shown to the executor.

    ScienceWorld exposes a free-form text command interface. The model often
    follows the high-level instruction too literally and focuses on the first
    visible noun in the room. For find-living-thing tasks this is dangerous:
    hallway observations often contain decoys such as drawings, beds, lights,
    paintings, or air, while the agent must navigate first to find an actually
    living object.

    These hints do not change the verifier or reward. They only make the valid
    action strategy explicit in the observation text.
    """
    instr_l = (instr or "").lower()
    obs_l = (sw_obs or "").lower()
    hints = [
        "Use ONE exact ScienceWorld command, not an explanation.",
        "Common commands include: open door to <room>, go to <room>, focus on <visible object>, take <visible object>, put <object> in <box>.",
        "Choose objects that are explicitly visible in the OBSERVATION. Do not invent objects.",
    ]
    if "living thing" in instr_l:
        hints.extend([
            "For find-living-thing: drawings, paintings, beds, air, light bulbs, boxes, and furniture are NOT living things.",
            "If the current room does not visibly contain a living thing, do NOT focus on a non-living object; navigate first.",
            "Prefer searching rooms such as greenhouse, kitchen, living room, bedroom, or workshop by opening a door and going there.",
            "Only use 'focus on <object>' after a plausible living thing is visible in the current observation.",
        ])
        if "greenhouse" in obs_l:
            hints.append("A greenhouse is a good place to search for living things; consider opening/go to greenhouse before focusing on hallway decoys.")
    return "\n".join(f"  - {h}" for h in hints)


class ScienceWorldEnv(AgenticEnv):
    def __init__(self, seed: int = 0):
        self._seed = int(seed)
        self._counter = 0
        self._sw = None                                  # lazy java env
        self._task: Optional[Task] = None
        self._preds: List[str] = []                      # predicate ids (sg{idx})
        self._pred_desc: Dict[str, str] = {}
        self._prev_done: Dict[str, bool] = {}            # last-known per-subgoal completion
        self._score: float = 0.0
        self._done = False
        self._steps = 0

    # ------------------------------------------------------------------ lazy java env
    def _engine(self):
        if self._sw is None:
            from scienceworld import ScienceWorldEnv as _SW
            self._sw = _SW("", envStepLimit=_STEP_LIMIT)
        return self._sw

    def _level_for(self, difficulty: float) -> str:
        d = max(0.0, min(1.0, float(difficulty)))
        return _LADDER[int(round(d * (len(_LADDER) - 1)))]

    # ------------------------------------------------------------------ contract
    def sample_new_task(self, rng: Any, difficulty: float) -> Task:
        r = rng if isinstance(rng, random.Random) else random.Random(self._seed + self._counter)
        task_name = self._level_for(difficulty)
        sw = self._engine()
        names = set(sw.get_task_names())
        if task_name not in names:                       # robust fallback
            task_name = "boil" if "boil" in names else sorted(names)[0]
        sw.load(task_name, 0, "")
        train_vars = sw.get_variations_train()
        var = r.choice(train_vars) if train_vars else 0
        sw.load(task_name, var, "")
        desc = sw.get_task_description()
        self._counter += 1
        return Task(
            task_id=f"sciworld_{task_name}_v{var}_{self._counter}",
            spec={"task_name": task_name, "variation": var, "instruction": desc},
            constraints={"task_name": task_name, "variation": var},
            info={"difficulty": round(float(difficulty), 3), "task_name": task_name},
        )

    def list_tasks(self, split: str) -> List[Task]:
        diffs = [0.2, 0.5, 0.8] if split == "train" else [0.5]
        return [self.sample_new_task(random.Random(self._seed + i), d) for i, d in enumerate(diffs)]

    def predicate_library(self, task: Task) -> List[str]:
        # resolve subgoals by loading the task (cheap; java env reused)
        sw = self._engine()
        sw.load(task.constraints["task_name"], int(task.constraints.get("variation", 0)), "")
        sw.reset()
        sgs = _parse_goal_progress(sw.get_goal_progress())
        return [g["pid"] for g in sgs]

    def is_well_posed(self, task: Task) -> bool:
        try:
            return len(self.predicate_library(task)) > 0
        except Exception:
            return False

    def reset(self, task: Task) -> Observation:
        self._task = task
        sw = self._engine()
        sw.load(task.constraints["task_name"], int(task.constraints.get("variation", 0)), "")
        obs, info = sw.reset()
        sgs = _parse_goal_progress(sw.get_goal_progress())
        self._preds = [g["pid"] for g in sgs]
        self._pred_desc = {g["pid"]: g["desc"] for g in sgs}
        self._prev_done = {g["pid"]: g["done"] for g in sgs}
        self._score = float(info.get("score", 0) or 0)
        self._done = False
        self._steps = 0
        return self._make_obs(obs)

    def _make_obs(self, sw_obs: str) -> Observation:
        instr = self._task.spec.get("instruction", "") if self._task else ""
        sw = self._engine()
        inv = ""
        try:
            inv = sw.inventory()
        except Exception:
            pass
        # surface the sub-goal checklist so the model can see what's left (and the order)
        checklist = "\n".join(
            f"  [{'x' if self._prev_done.get(pid) else ' '}] {self._pred_desc.get(pid,'')}"
            for pid in self._preds
        )
        guidance = _scienceworld_command_guidance(instr, sw_obs)
        text = (f"TASK: {instr}\n\nSUB-GOALS (in order):\n{checklist}\n\n"
                f"OBSERVATION:\n{sw_obs}\n\n{inv}\n"
                f"ACTION GUIDANCE:\n{guidance}\n\n"
                f"Reply with ONE ScienceWorld command (e.g. 'open door to kitchen', "
                f"'go to kitchen', 'focus on water', 'activate stove').")
        return Observation(text=text, available_actions=[
            "open door to <room>",
            "go to <room>",
            "focus on <visible object>",
            "take <visible object>",
            "put <object> in <box>",
            "activate <object>",
            "<ScienceWorld command>",
        ],
                           state={"score": self._score, "task": self._task.task_id if self._task else ""})

    def step(self, action: Action) -> StepResult:
        if self._task is None:
            raise RuntimeError("step() before reset()")
        sw = self._engine()
        cmd = (action.parsed or {}).get("cmd") or _parse_action(action.raw)
        obs, reward, done, info = sw.step(cmd)
        self._steps += 1
        new_score = float(info.get("score", 0) or 0)
        # invalid command (ScienceWorld echoes a no-op message and score is unchanged)
        invalid = bool(re.search(r"no known action|ambiguous|not sure what|can't|cannot", obs, re.I))

        # sub-goal deltas -> StageRecords (the trajectory-level signal)
        sgs = _parse_goal_progress(sw.get_goal_progress())
        records: List[StageRecord] = []
        cur_done = {g["pid"]: g["done"] for g in sgs}
        for pid, now in cur_done.items():
            was = self._prev_done.get(pid, False)
            if now != was:
                records.append(StageRecord(
                    predicate_id=pid, before="sat" if was else "unsat",
                    after="sat" if now else "unsat",
                    failure_type="none", char_span=(0, 0),
                    info={"desc": self._pred_desc.get(pid, ""), "recovered": (was is False and now)}))
        rinfo: Dict[str, Any] = {"score": new_score, "cmd": cmd}
        if invalid:
            rinfo["failure_type"] = "format"
            rinfo["needs_replan"] = True
            records.append(StageRecord(predicate_id="action", before="unsat", after="unsat",
                                       failure_type="format", char_span=(0, 0), info={"cmd": cmd}))
        elif new_score < self._score:                    # regressed a sub-goal -> recovery opportunity
            rinfo["failure_type"] = "constraint_violation"
            rinfo["needs_replan"] = True

        self._prev_done = cur_done
        self._score = new_score
        score_done = bool(done) or new_score >= 100 or self._steps >= _STEP_LIMIT
        if score_done:
            self._done = True
        tr = self.verify_final() if score_done else None
        return StepResult(obs=self._make_obs(obs), stage_records=records, done=score_done,
                          terminal_reward=tr, info=rinfo)

    def verify_final(self) -> float:
        return max(0.0, min(1.0, self._score / 100.0))

    def needs_replan(self, last: StepResult) -> bool:
        return bool(last.info.get("needs_replan", False)) if last else False
