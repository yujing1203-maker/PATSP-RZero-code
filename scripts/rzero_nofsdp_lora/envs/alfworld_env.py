# -*- coding: utf-8 -*-
"""ALFWorld adapter -> envs.base.AgenticEnv  (benchmark #3 for PATSP).

ALFWorld (Shridhar et al. 2021) is the canonical text embodied-agent benchmark: household
tasks ("put a clean mug in the coffee machine") over TextWorld. Long-horizon with hard
PREREQUISITE structure -- you must FIND, then TAKE, then (clean/heat/cool), then PLACE, in
that order -- so subgoal-ordering / premature-stop / missing-prerequisite failures + recovery
are first-class (the PATSP signals).

ALFWorld's native reward is sparse (won=1.0). We add a milestone sub-goal decomposition per
task type (take -> transform -> place), detected from the game feedback, so verify_final is a
continuous partial-credit signal and StageRecords carry the per-subgoal events PATSP trains on.

Install: `pip install --no-deps alfworld textworld`; `alfworld-download` (sets ALFWORLD_DATA).
Data layout used: $ALFWORLD_DATA/json_2.1.1/{train,valid_seen,valid_unseen} + logic/alfred.*.
Heavy TextWorld env is created lazily on first reset and reused across episodes.
"""

from __future__ import annotations

import os
import re
import random
from typing import Any, Dict, List, Optional

from envs.base import Task, Observation, Action, StepResult, AgenticEnv
from verifiers.base import StageRecord

_RZ_STORAGE = os.environ.get("RZERO_STORAGE", os.path.expanduser("~/RZero_storage"))
_ALFWORLD_DATA = os.environ.get("ALFWORLD_DATA", os.path.join(_RZ_STORAGE, "alfworld_data"))
_MAX_STEPS = int(os.environ.get("ALFWORLD_STEP_LIMIT", "40"))

# The 6 ALFWorld task types, ordered easy -> hard (used for the difficulty ladder).
_TASK_TYPES = [
    "pick_and_place_simple",                 # take -> place
    "look_at_obj_in_light",                  # take -> turn on lamp
    "pick_two_obj_and_place",                # take -> place -> take -> place
    "pick_clean_then_place_in_recep",        # take -> clean -> place
    "pick_heat_then_place_in_recep",         # take -> heat -> place
    "pick_cool_then_place_in_recep",         # take -> cool -> place
]
# milestone sub-goals per task family, detected from feedback keywords.
_MILESTONES = {
    "pick_and_place_simple":          [("hold", "take"), ("place", "put")],
    "look_at_obj_in_light":           [("hold", "take"), ("light", "turn on")],
    "pick_two_obj_and_place":         [("hold", "take"), ("place", "put"), ("hold2", "take"), ("place2", "put")],
    "pick_clean_then_place_in_recep": [("hold", "take"), ("clean", "clean"), ("place", "put")],
    "pick_heat_then_place_in_recep":  [("hold", "take"), ("heat", "heat"), ("place", "put")],
    "pick_cool_then_place_in_recep":  [("hold", "take"), ("cool", "cool"), ("place", "put")],
}


def _task_family(gamefile: str) -> str:
    base = gamefile.replace("\\", "/")
    for tt in _TASK_TYPES:
        if tt in base:
            return tt
    # the dir name prefix (e.g. pick_clean_then_place_in_recep-Tomato-...) -> match by prefix
    name = base.split("/json_2.1.1/")[-1]
    for tt in _TASK_TYPES:
        if name.startswith(tt):
            return tt
    return "pick_and_place_simple"


def _parse_cmd(raw: str) -> str:
    t = (raw or "").strip()
    m = re.search(r"<action>(.*?)</action>", t, flags=re.S)
    if m:
        t = m.group(1).strip()
    line = next((ln.strip() for ln in t.splitlines() if ln.strip()), t)
    return re.sub(r"^(action|command)\s*[:\-]\s*", "", line, flags=re.I).strip()[:200]


class ALFWorldEnv(AgenticEnv):
    def __init__(self, seed: int = 0):
        self._seed = int(seed)
        self._counter = 0
        self._tw = None                     # batched TextWorld env (lazy)
        self._task: Optional[Task] = None
        self._family = "pick_and_place_simple"
        self._milestones: List[tuple] = []
        self._done_ms: Dict[str, bool] = {}
        self._goal = ""
        self._admissible: List[str] = []
        self._won = False
        self._steps = 0
        self._last_obs = ""

    # ------------------------------------------------------------------ lazy TW env
    def _config(self) -> Dict[str, Any]:
        root = _ALFWORLD_DATA
        j = os.path.join(root, "json_2.1.1")
        return {
            "dataset": {"data_path": os.path.join(j, "train"),
                        "eval_id_data_path": os.path.join(j, "valid_seen"),
                        "eval_ood_data_path": os.path.join(j, "valid_unseen"),
                        "num_train_games": 0, "num_eval_games": 0},
            "logic": {"domain": os.path.join(root, "logic", "alfred.pddl"),
                      "grammar": os.path.join(root, "logic", "alfred.twl2")},
            "env": {"task_types": [1, 2, 3, 4, 5, 6], "goal_desc_human_anns_prob": 0.0,
                    "domain_randomization": False, "expert_type": "handcoded",
                    "regen_game_files": False},
            "general": {"training_method": "dagger", "random_seed": self._seed},
            "rl": {"training": {"max_nb_steps_per_episode": _MAX_STEPS}},
            "dagger": {"training": {"max_nb_steps_per_episode": _MAX_STEPS}},
        }

    def _engine(self):
        if self._tw is None:
            from alfworld.agents.environment import get_environment
            AlfredTWEnv = get_environment("AlfredTWEnv")
            base = AlfredTWEnv(self._config(), train_eval=os.environ.get("ALFWORLD_SPLIT", "train"))
            self._tw = base.init_env(batch_size=1)
        return self._tw

    def _level_for(self, difficulty: float) -> str:
        d = max(0.0, min(1.0, float(difficulty)))
        return _TASK_TYPES[int(round(d * (len(_TASK_TYPES) - 1)))]

    # ------------------------------------------------------------------ contract
    def sample_new_task(self, rng: Any, difficulty: float) -> Task:
        # ALFWorld samples a game on reset(); we record the target family + a seed so the
        # episode keeps sampling until it draws the requested task family (bounded tries).
        self._counter += 1
        fam = self._level_for(difficulty)
        seed = (rng.randint(0, 1 << 30) if isinstance(rng, random.Random)
                else (self._seed + self._counter))
        return Task(
            task_id=f"alfworld_{fam}_{self._counter}",
            spec={"family": fam, "instruction": f"ALFWorld {fam} task"},
            constraints={"family": fam, "seed": seed},
            info={"difficulty": round(float(difficulty), 3), "family": fam},
        )

    def list_tasks(self, split: str) -> List[Task]:
        diffs = [0.0, 0.3, 0.6] if split == "train" else [0.0]
        return [self.sample_new_task(random.Random(self._seed + i), d) for i, d in enumerate(diffs)]

    def predicate_library(self, task: Task) -> List[str]:
        fam = task.constraints.get("family", "pick_and_place_simple")
        return [f"ms_{k}" for k, _ in _MILESTONES.get(fam, _MILESTONES["pick_and_place_simple"])]

    def is_well_posed(self, task: Task) -> bool:
        return len(self.predicate_library(task)) > 0

    def reset(self, task: Task) -> Observation:
        self._task = task
        tw = self._engine()
        target = task.constraints.get("family", "pick_and_place_simple")
        obs, info = tw.reset()
        gamefiles = info.get("extra.gamefile", [""])
        # sample up to N games to land on the requested family (keeps difficulty meaningful)
        for _ in range(12):
            gf = (gamefiles[0] if isinstance(gamefiles, list) else gamefiles) or ""
            if _task_family(gf) == target:
                break
            obs, info = tw.reset()
            gamefiles = info.get("extra.gamefile", [""])
        gf = (gamefiles[0] if isinstance(gamefiles, list) else gamefiles) or ""
        self._family = _task_family(gf)
        self._milestones = _MILESTONES.get(self._family, _MILESTONES["pick_and_place_simple"])
        self._done_ms = {f"ms_{k}": False for k, _ in self._milestones}
        ob0 = obs[0] if isinstance(obs, (list, tuple)) else obs
        self._goal = self._extract_goal(ob0)
        self._admissible = self._adm(info)
        self._won = False
        self._steps = 0
        self._last_obs = ob0
        return self._make_obs(ob0)

    @staticmethod
    def _extract_goal(obs: str) -> str:
        m = re.search(r"Your task is to:\s*(.+)", obs)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _adm(info: dict) -> List[str]:
        a = info.get("admissible_commands")
        if isinstance(a, list) and a and isinstance(a[0], list):
            return a[0]
        return a or []

    def _make_obs(self, sw_obs: str) -> Observation:
        checklist = "\n".join(
            f"  [{'x' if self._done_ms.get('ms_'+k) else ' '}] {k}" for k, _ in self._milestones)
        adm = self._admissible[:30]
        text = (f"TASK: {self._goal}\n\nSUB-GOALS (in order):\n{checklist}\n\n"
                f"OBSERVATION:\n{sw_obs}\n\n"
                f"ADMISSIBLE ACTIONS (examples): {', '.join(adm[:20])}\n"
                f"Reply with ONE action (e.g. 'go to fridge 1', 'take mug 1 from countertop 1', "
                f"'clean mug 1 with sinkbasin 1', 'put mug 1 in/on coffeemachine 1').")
        return Observation(text=text, available_actions=adm,
                           state={"family": self._family, "won": self._won})

    def _detect(self, feedback: str) -> List[str]:
        """Return milestone keys newly satisfied per this step's feedback."""
        fb = feedback.lower()
        fired = []
        for k, verb in self._milestones:
            if self._done_ms.get(f"ms_{k}"):
                continue
            ok = False
            if k.startswith("hold") and ("you pick up" in fb or "you take" in fb or "now carrying" in fb):
                ok = True
            elif k.startswith("place") and ("you put" in fb or "you place" in fb or "you move" in fb):
                ok = True
            elif k == "clean" and ("you clean" in fb or "is now clean" in fb):
                ok = True
            elif k == "heat" and ("you heat" in fb or "is now hot" in fb):
                ok = True
            elif k == "cool" and ("you cool" in fb or "is now cool" in fb):
                ok = True
            elif k == "light" and ("you turn on" in fb or "is now on" in fb):
                ok = True
            if ok:
                fired.append(k)
        return fired

    def step(self, action: Action) -> StepResult:
        if self._task is None:
            raise RuntimeError("step() before reset()")
        tw = self._engine()
        cmd = _parse_cmd(action.raw)
        obs, score, done, info = tw.step([cmd])
        ob = obs[0] if isinstance(obs, (list, tuple)) else obs
        d0 = (done[0] if isinstance(done, (list, tuple)) else done)
        won = bool((info.get("won") or [False])[0] if isinstance(info.get("won"), list) else info.get("won"))
        self._won = self._won or won
        self._admissible = self._adm(info)
        self._steps += 1
        records: List[StageRecord] = []
        rinfo: Dict[str, Any] = {"cmd": cmd, "won": self._won}

        invalid = bool(re.search(r"nothing happens|can't|cannot|not able", ob, re.I))
        if invalid:
            rinfo["failure_type"] = "missing"          # likely a prerequisite not met / bad target
            rinfo["needs_replan"] = True
            records.append(StageRecord(predicate_id="action", before="unsat", after="unsat",
                                       failure_type="missing", char_span=(0, 0), info={"cmd": cmd}))
        for k in self._detect(ob):
            pid = f"ms_{k}"
            records.append(StageRecord(predicate_id=pid, before="unsat", after="sat",
                                       failure_type="none", char_span=(0, 0), info={}))
            self._done_ms[pid] = True

        terminal = bool(d0) or self._won or self._steps >= _MAX_STEPS
        if terminal:
            self._task and None
        tr = self.verify_final() if terminal else None
        self._last_obs = ob
        return StepResult(obs=self._make_obs(ob), stage_records=records, done=terminal,
                          terminal_reward=tr, info=rinfo)

    def verify_final(self) -> float:
        if self._won:
            return 1.0
        ms = self._milestones or [("x", "")]
        return sum(1 for k, _ in ms if self._done_ms.get(f"ms_{k}")) / float(len(ms))

    def needs_replan(self, last: StepResult) -> bool:
        return bool(last.info.get("needs_replan", False)) if last else False
