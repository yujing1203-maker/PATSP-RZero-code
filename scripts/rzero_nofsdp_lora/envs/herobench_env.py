# -*- coding: utf-8 -*-
"""HeroBench adapter -> envs.base.AgenticEnv  (benchmark #4 for PATSP).

HeroBench (stefanrer/HeroBench, based on ArtifactsMMO) is an RPG world with DEEP directed
crafting chains + skill-level prerequisites + locations. It is the richest dependency-long-horizon
substrate for PATSP: to craft a target you must gather base resources (at the right tile, with the
right gather-skill level), craft intermediates (at the right workshop tile, with the right craft-skill
level), in dependency order. The env exposes EXACTLY the PATSP failure modes:
  - craft fails (HTTP 500) with structured booleans on_workshop_tile / needed_skill_level /
    enough_items_for_craft  -> wrong-location / skill-prereq / missing-ingredient failures.
  - gather fails (493 skill too low / 598 not on tile).

Architecture: HeroBench is a FastAPI server (we run the SQLite backend on a per-instance port) +
this adapter as a thin requests client. We boot the server lazily and reuse it across episodes; each
reset() creates a FRESH character so inventory/skills start clean.

Sub-goals (predicate_library) come from the task's crafting_tree (post-order DFS): each craftable
node -> `craft_<code>`, each gatherable basic leaf -> `gather_<code>`. verify_final reproduces
HeroBench's progress score = (# distinct sub-goals achieved) / (# sub-goals in the optimal plan),
all weight 1, deduped (see scoring_pipeline.py / utils.py compute_*_episode_reward).

Action grammar (parsed from Action.raw):
    goto <code>        -- move to the tile holding resource/workshop <code> (resolved via /maps)
    move <x> <y>       -- move to coordinates
    gather             -- gather at the current tile
    craft <code> [n]   -- craft item <code> (qty n, default 1)
    fight              -- fight the monster on the current tile
    done               -- stop
"""

from __future__ import annotations

import json
import os
import re
import time
import socket
import subprocess
from typing import Any, Dict, List, Optional

import requests

from envs.base import Task, Observation, Action, StepResult, AgenticEnv
from verifiers.base import StageRecord

# Project root derived from THIS file's location (.../<PROJ>/scripts/rzero_nofsdp_lora/envs/herobench_env.py)
# so the project is self-contained / portable -- no hardcoded user path. RZERO_PROJECT overrides.
_PROJ = os.environ.get(
    "RZERO_PROJECT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)
_HB_ROOT = os.environ.get("HEROBENCH_ROOT", os.path.join(_PROJ, "third_party", "HeroBench"))
_SQL_DIR = os.path.join(_HB_ROOT, "Virtual_Environment", "FastApi_SQLite_Ver")
_DATASET = {
    "base": os.path.join(_HB_ROOT, "datasets", "dataset_tasks.json"),
    "leveling": os.path.join(_HB_ROOT, "datasets", "dataset_tasks_leveling.json"),
    "noise": os.path.join(_HB_ROOT, "datasets", "dataset_tasks_noise_leveling.json"),
}
_GATHER_SKILLS = {"mining", "woodcutting", "fishing"}


# --------------------------------------------------------------------------- crafting-tree parsing
def _walk_subgoals(node: Dict[str, Any], crafts: List[Dict], gathers: List[Dict]) -> None:
    """Post-order DFS: collect craft nodes (ingredients-before-parent) and basic gatherable leaves."""
    if not isinstance(node, dict):
        return
    if "craft" in node and node.get("craft"):
        for ing in node["craft"].get("ingredients", []):
            _walk_subgoals(ing, crafts, gathers)
        crafts.append({"code": node["code"], "skill": node["craft"].get("skill"),
                       "level": node["craft"].get("level", 1),
                       "ingredients": [(i["code"], i.get("required_quantity", 1))
                                       for i in node["craft"].get("ingredients", [])]})
    elif node.get("basic"):
        if node.get("resources"):
            res = node["resources"][0]
            locs = res.get("locations", [])
            gathers.append({"code": node["code"], "skill": res.get("skill"),
                            "level": res.get("resource_level", 1),
                            "loc": (locs[0]["x"], locs[0]["y"]) if locs else None})
        # mob-drop / pure-gatherable leaves are not gather-scored here (kept simple)


class HeroBenchEnv(AgenticEnv):
    def __init__(self, seed: int = 0):
        self._seed = int(seed)
        self._counter = 0
        self._proc = None
        self._port = int(os.environ.get("HEROBENCH_PORT", "8000"))
        self._base = f"http://127.0.0.1:{self._port}"
        self._tasks_by_level: Dict[str, List[dict]] = {}
        # episode state
        self._task: Optional[Task] = None
        self._cname = ""
        self._target = ""
        self._crafts: List[Dict] = []
        self._gathers: List[Dict] = []
        self._preds: List[str] = []
        self._done_preds: set = set()
        self._maps_cache: Dict[str, tuple] = {}
        self._goto_alias: Dict[str, str] = {}
        self._steps = 0
        self._done = False

    # ------------------------------------------------------------------ server lifecycle
    def _port_open(self) -> bool:
        try:
            requests.get(f"{self._base}/docs", timeout=2)
            return True
        except Exception:
            return False

    def _engine(self):
        if self._port_open():
            return
        # boot the SQLite FastAPI server on our port (loads world on startup ~35s)
        env = dict(os.environ)
        # Per-port DB so parallel arms (R-Zero / PATSP) never share or rm_db-wipe one
        # another's world. Server cwd is _SQL_DIR, so this relative file lands there.
        env.setdefault("HEROBENCH_DB", f"artifact_{self._port}.db")
        log = open(os.environ.get("HEROBENCH_SERVER_LOG", f"/tmp/herobench_server_{self._port}.log"), "w")
        self._proc = subprocess.Popen(
            ["python", "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(self._port)],
            cwd=_SQL_DIR, stdout=log, stderr=subprocess.STDOUT, env=env)
        for _ in range(120):
            if self._port_open():
                return
            time.sleep(2)
        raise RuntimeError(f"HeroBench server failed to start on :{self._port}")

    # ------------------------------------------------------------------ http helpers
    def _get(self, path: str, **params):
        try:
            r = requests.get(f"{self._base}{path}", params={k: v for k, v in params.items() if v is not None}, timeout=30)
            return r.status_code, (r.json() if r.content else {})
        except Exception as e:
            return -1, {"error": str(e)}

    def _post(self, path: str, payload):
        try:
            r = requests.post(f"{self._base}{path}", json=payload, timeout=60,
                              headers={"Content-Type": "application/json", "Accept": "application/json"})
            return r.status_code, (r.json() if r.content else {})
        except Exception as e:
            return -1, {"error": str(e)}

    def _char(self) -> Dict[str, Any]:
        _, d = self._get(f"/characters/{self._cname}")
        return d if isinstance(d, dict) else {}

    def _find_tile(self, code: str) -> Optional[tuple]:
        if code in self._maps_cache:
            return self._maps_cache[code]
        st, d = self._get("/maps", content_code=code)
        loc = None
        if st == 200 and isinstance(d, list) and d:
            loc = (d[0]["x"], d[0]["y"])
        elif st == 200 and isinstance(d, dict) and d.get("maps"):
            m = d["maps"][0]; loc = (m["x"], m["y"])
        self._maps_cache[code] = loc
        return loc

    # ------------------------------------------------------------------ tasks
    def _load_dataset(self):
        if self._tasks_by_level:
            return
        variant = os.environ.get("HEROBENCH_VARIANT", "base")
        path = _DATASET.get(variant, _DATASET["base"])
        data = json.load(open(path))
        # keep only Type-A crafting tasks (have crafting_tree); PATSP targets the crafting chain
        for lvl, lst in data.items():
            keep = [t for t in lst if isinstance(t, dict) and t.get("crafting_tree")]
            if keep:
                self._tasks_by_level[lvl] = keep

    def _level_for(self, difficulty: float) -> str:
        self._load_dataset()
        lvls = sorted(self._tasks_by_level.keys(), key=lambda s: int(s))
        d = max(0.0, min(1.0, float(difficulty)))
        return lvls[int(round(d * (len(lvls) - 1)))]

    def sample_new_task(self, rng: Any, difficulty: float) -> Task:
        import random as _r
        r = rng if isinstance(rng, _r.Random) else _r.Random(self._seed + self._counter)
        lvl = self._level_for(difficulty)
        pool = self._tasks_by_level[lvl]
        # Held-out eval tasks (by pool index) are excluded from training/challenger sampling so the
        # frozen D2 set stays disjoint from what the solver trains on (HEROBENCH_TRAIN_EXCLUDE="0,4,9,11").
        excl = os.environ.get("HEROBENCH_TRAIN_EXCLUDE", "")
        if excl:
            ex = {int(x) for x in excl.replace(" ", "").split(",") if x.lstrip("-").isdigit()}
            pool = [t for i, t in enumerate(pool) if i not in ex] or pool
        spec_task = r.choice(pool)
        self._counter += 1
        ct = spec_task["crafting_tree"]
        return Task(
            task_id=f"herobench_L{lvl}_{ct['code']}_{self._counter}",
            spec={"item": spec_task.get("item"), "target": ct["code"], "crafting_tree": ct,
                  "instruction": f"Craft '{spec_task.get('item')}' (code={ct['code']})."},
            constraints={"target": ct["code"], "crafting_tree": ct, "level": lvl},
            info={"difficulty": round(float(difficulty), 3), "level": lvl,
                  "total_difficulty": spec_task.get("total_difficulty")},
        )

    def _collect_goto_aliases(self, node: Dict[str, Any]) -> None:
        """Map task-visible HeroBench location/resource names to backend /maps content_code.

        The task spec often exposes location["skin"] such as "forest_goldore1",
        while the FastAPI /maps endpoint expects content_code such as "gold_rocks".
        This alias table lets actions like `goto forest_goldore1` resolve correctly.
        """
        if not isinstance(node, dict):
            return

        node_code = node.get("code")
        for res in node.get("resources", []) or []:
            if not isinstance(res, dict):
                continue
            resource_code = res.get("resource_code")
            for loc in res.get("locations", []) or []:
                if not isinstance(loc, dict):
                    continue
                content = loc.get("content") or {}
                content_code = content.get("code") or resource_code
                if not content_code:
                    continue
                for alias in (loc.get("skin"), resource_code, node_code):
                    if alias:
                        self._goto_alias[str(alias).lower()] = str(content_code).lower()

        craft = node.get("craft") or {}
        for ing in craft.get("ingredients", []) or []:
            self._collect_goto_aliases(ing)

    def list_tasks(self, split: str) -> List[Task]:
        import random as _r
        diffs = [0.2, 0.4, 0.6] if split == "train" else [0.4]
        return [self.sample_new_task(_r.Random(self._seed + i), d) for i, d in enumerate(diffs)]

    def _compute_subgoals(self, ct: Dict[str, Any]):
        crafts, gathers = [], []
        _walk_subgoals(ct, crafts, gathers)
        self._goto_alias = {}
        self._collect_goto_aliases(ct)
        self._crafts, self._gathers = crafts, gathers
        # predicate library: gather leaves then craft nodes (build order), each weight 1 (HeroBench-style)
        self._preds = [f"gather_{g['code']}" for g in gathers] + [f"craft_{c['code']}" for c in crafts]

    def predicate_library(self, task: Task) -> List[str]:
        self._compute_subgoals(task.constraints["crafting_tree"])
        return list(self._preds)

    def is_well_posed(self, task: Task) -> bool:
        try:
            self._compute_subgoals(task.constraints["crafting_tree"])
            return len(self._preds) > 0
        except Exception:
            return False

    # ------------------------------------------------------------------ episode
    def reset(self, task: Task) -> Observation:
        self._engine()
        self._task = task
        self._target = task.constraints["target"]
        self._compute_subgoals(task.constraints["crafting_tree"])
        self._done_preds = set()
        self._maps_cache = {}
        self._steps = 0
        self._done = False
        # fresh character per episode
        self._counter += 1
        self._cname = f"hb_{os.getpid()}_{self._counter}"
        self._post("/characters/create", {"name": self._cname, "skin": "men1"})
        return self._make_obs("New character created at (0,0). Plan the crafting chain.")

    def _make_obs(self, last_feedback: str) -> Observation:
        ch = self._char()
        inv = ", ".join(f"{s['code']}x{s['quantity']}" for s in ch.get("inventory", [])) or "(empty)"
        pos = f"({ch.get('x',0)},{ch.get('y',0)})"
        # recipe / plan lines with locations + skill reqs
        plan = []
        for g in self._gathers:
            mark = "x" if f"gather_{g['code']}" in self._done_preds else " "
            goto_code = self._goto_alias.get(g['code'], g['code'])
            plan.append(f"  [{mark}] gather {g['code']} @ loc{g['loc']} via `goto {goto_code}` then `gather` (skill {g['skill']} lv{g['level']})")
        for c in self._crafts:
            mark = "x" if f"craft_{c['code']}" in self._done_preds else " "
            ings = " + ".join(f"{co}x{q}" for co, q in c["ingredients"])
            plan.append(f"  [{mark}] craft {c['code']} <= {ings} (workshop {c['skill']} lv{c['level']})")
        text = (f"GOAL: craft '{self._target}'.\n\nSUB-GOALS / RECIPE (in build order):\n" + "\n".join(plan) +
                f"\n\nCHARACTER pos={pos}  inventory: {inv}\n"
                f"LAST: {last_feedback}\n\n"
                f"Actions: `goto <code>` (move to a resource/workshop tile) | `gather` | "
                f"`craft <code> [qty]` | `unequip <slot>` (e.g. weapon) | `fight` | `done`")
        return Observation(text=text, available_actions=["goto <code>", "gather", "craft <code>", "fight", "done"],
                           state={"pos": pos, "target": self._target})

    @staticmethod
    def _parse(raw: str) -> Dict[str, Any]:
        t = (raw or "").strip().lower()
        m = re.search(r"<action>(.*?)</action>", t, flags=re.S)
        if m:
            t = m.group(1).strip()
        t = next((ln.strip() for ln in t.splitlines() if ln.strip()), t)
        if re.search(r"\b(done|finish|stop)\b", t):
            return {"op": "done"}
        m = re.search(r"\b(goto|go to|move to)\s+([a-z0-9_]+)\b", t)
        if m:
            return {"op": "goto", "code": m.group(2)}
        m = re.search(r"\bmove\s+(-?\d+)\s+(-?\d+)", t)
        if m:
            return {"op": "move", "x": int(m.group(1)), "y": int(m.group(2))}
        m = re.search(r"\bunequip\s+([a-z0-9_]+)", t)
        if m:
            return {"op": "unequip", "slot": m.group(1)}
        if re.search(r"\bgather|mine|chop|fish\b", t):
            return {"op": "gather"}
        m = re.search(r"\b(craft|make|create)\s+([a-z0-9_]+)(?:\s+(\d+))?", t)
        if m:
            return {"op": "craft", "code": m.group(2), "qty": int(m.group(3) or 1)}
        if re.search(r"\bfight|attack|kill\b", t):
            return {"op": "fight"}
        return {"op": "unknown"}

    def step(self, action: Action) -> StepResult:
        if self._task is None:
            raise RuntimeError("step() before reset()")
        p = self._parse(action.raw)
        op = p.get("op")
        self._steps += 1
        records: List[StageRecord] = []
        info: Dict[str, Any] = {"op": op}
        fb = ""

        if op == "done":
            self._done = True
            return StepResult(obs=self._make_obs("done"), stage_records=[], done=True,
                              terminal_reward=self.verify_final(), info={"reached_done": True})

        if op == "goto":
            raw_code = p["code"]
            code = self._goto_alias.get(raw_code, raw_code)
            loc = self._find_tile(code)
            info["goto_raw_code"] = raw_code
            info["goto_resolved_code"] = code
            if loc is None:
                info["failure_type"] = "format"; fb = f"no tile for {raw_code} -> {code}"
            else:
                st, d = self._post(f"/my/{self._cname}/action/move", {"x": loc[0], "y": loc[1]})
                fb = f"move->{loc} via {raw_code}->{code} status {st}"
        elif op == "move":
            st, d = self._post(f"/my/{self._cname}/action/move", {"x": p["x"], "y": p["y"]})
            fb = f"move->({p['x']},{p['y']}) status {st}"
        elif op == "gather":
            st, d = self._post(f"/my/{self._cname}/action/gathering", 1)
            if st == 200:
                # which resource? infer from items returned
                got = [it.get("code") for it in d.get("details", {}).get("items", [])] if isinstance(d, dict) else []
                for g in self._gathers:
                    pid = f"gather_{g['code']}"
                    if g["code"] in got and pid not in self._done_preds:
                        self._done_preds.add(pid)
                        records.append(StageRecord(predicate_id=pid, before="unsat", after="sat",
                                                   failure_type="none", char_span=(0, 0), info={}))
                fb = f"gathered {got}"
            else:
                ft = "missing" if st == 493 else ("constraint_violation" if st == 598 else "format")
                info["failure_type"] = ft; info["needs_replan"] = True
                records.append(StageRecord(predicate_id="gather", before="unsat", after="unsat",
                                           failure_type=ft, char_span=(0, 0), info={"status": st}))
                fb = f"gather failed status {st}"
        elif op == "craft":
            code, qty = p["code"], p.get("qty", 1)
            st, d = self._post(f"/my/{self._cname}/action/crafting", {"code": code, "quantity": qty})
            pid = f"craft_{code}"
            if st == 200:
                if pid not in self._done_preds:
                    self._done_preds.add(pid)
                    records.append(StageRecord(predicate_id=pid, before="unsat", after="sat",
                                               failure_type="none", char_span=(0, 0), info={}))
                fb = f"crafted {code}"
            elif st == 500:
                # structured craft-failure booleans -> map to PATSP failure types
                errs = {}
                msg = (d.get("error", {}) or {}).get("message", {}) if isinstance(d, dict) else {}
                if isinstance(msg, dict):
                    errs = msg.get("errors", {})
                if errs.get("enough_items_for_craft") is False:
                    ft = "missing"               # missing ingredients (ordering/prereq) -- the PATSP signal
                elif errs.get("on_workshop_tile") is False:
                    ft = "constraint_violation"  # wrong location
                elif errs.get("needed_skill_level") is False:
                    ft = "missing"               # skill prereq not met
                else:
                    ft = "constraint_violation"
                info["failure_type"] = ft; info["needs_replan"] = True
                records.append(StageRecord(predicate_id=pid, before="unsat", after="unsat",
                                           failure_type=ft, char_span=(0, 0), info={"errors": errs}))
                fb = f"craft {code} failed: {errs}"
            else:
                info["failure_type"] = "format"
                records.append(StageRecord(predicate_id=pid, before="unsat", after="unsat",
                                           failure_type="format", char_span=(0, 0), info={"status": st}))
                fb = f"craft {code} invalid status {st}"
        elif op == "unequip":
            slot = p["slot"]
            slot = slot[:-5] if slot.endswith("_slot") else slot  # API wants the bare ItemSlot value
            st, d = self._post(f"/my/{self._cname}/action/unequip", {"slot": slot, "quantity": 1})
            fb = f"unequip {slot} status {st}"
        elif op == "fight":
            st, d = self._post(f"/my/{self._cname}/action/fight", None)
            fb = f"fight status {st}"
        else:
            info["failure_type"] = "format"; fb = "unparseable action"

        # success = target crafted
        target_done = f"craft_{self._target}" in self._done_preds
        done = target_done or self._steps >= int(os.environ.get("HEROBENCH_MAX_STEPS", "40"))
        if done:
            self._done = True
        tr = self.verify_final() if done else None
        return StepResult(obs=self._make_obs(fb), stage_records=records, done=done,
                          terminal_reward=tr, info=info)

    def verify_final(self) -> float:
        # HeroBench progress = (# distinct sub-goals achieved) / (# sub-goals in optimal plan), weight 1
        n = len(self._preds)
        if n == 0:
            return 0.0
        if f"craft_{self._target}" in self._done_preds:
            return 1.0
        return len([p for p in self._done_preds if p in set(self._preds)]) / float(n)

    def needs_replan(self, last: StepResult) -> bool:
        return bool(last.info.get("needs_replan", False)) if last else False
