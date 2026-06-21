# -*- coding: utf-8 -*-
"""TextCraft: a text crafting environment with PREREQUISITE-DEPENDENT sub-goals.

Why this env (PATSP motivation)
-------------------------------
R-Zero-style task-level self-play sees a task as a scalar pass-rate. Long-horizon failures,
though, are *localized* mid-trajectory errors: crafting in the wrong order, stopping early,
or trying to craft before gathering ingredients. TextCraft makes those failures FIRST-CLASS:

  * To craft the TARGET you must craft its ingredients first, recursively, down to base
    materials you `get`. This is a real prerequisite DAG -> ordering / premature-stop /
    missing-ingredient errors actually occur and are programmatically detectable.
  * Each non-base item is a SUB-GOAL (predicate `have_<item>`). A `craft` that is missing an
    ingredient yields a StageRecord with failure_type="missing" -- the signal the PATSP
    challenger/credit modules consume (a "weakness" event), and the point a RECOVERY can happen
    (gather the missing ingredient, then re-craft).

Contract: implements envs.base.AgenticEnv exactly. Pure-python, no torch/vllm.

Action grammar (parsed from Action.raw, case-insensitive, robust to surrounding prose):
    get <item>            -- obtain one unit of a BASE item (raw material)
    craft <item>          -- craft <item> if all its recipe ingredients are in inventory
    done / finalize       -- stop early (verify_final scores partial credit)

Difficulty -> recipe-DAG depth (longer dependency chain = longer horizon):
    depth = 1 + round(difficulty * 4)   # d=0 -> depth1 (trivial), d=1 -> depth5 (deep chain)

verify_final = 1.0 if the target was crafted, else fraction of sub-goal items obtained
(continuous partial credit -> a real learning signal, never trivially 1.0).
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

from envs.base import Task, Observation, Action, StepResult, AgenticEnv
from verifiers.base import StageRecord

# Minecraft-flavoured vocabulary so generated recipes read naturally.
_BASE_POOL = [
    "log", "stone", "iron_ore", "coal", "sand", "clay", "wheat", "leather",
    "feather", "flint", "gold_ore", "redstone", "string", "egg", "sugar_cane",
]
_CRAFT_POOL = [
    "plank", "stick", "torch", "furnace", "glass", "brick", "paper", "book",
    "iron_ingot", "gold_ingot", "tool_rod", "pickaxe", "axe", "shovel", "sword",
    "bucket", "shears", "compass", "clock", "cake", "bow", "arrow", "ladder",
    "chest", "sign", "bowl", "bread", "cookie", "armor_plate", "gear",
]


def _parse(raw: str) -> Dict[str, Any]:
    """raw executor text -> {'op': get|craft|done, 'item': <name>|None}."""
    t = (raw or "").strip().lower()
    # strip an <action>..</action> wrapper if the model emitted one
    m = re.search(r"<action>(.*?)</action>", t, flags=re.S)
    if m:
        t = m.group(1).strip()
    if re.search(r"\b(done|finalize|stop|finish)\b", t):
        return {"op": "done", "item": None}
    m = re.search(r"\b(get|gather|obtain|mine|collect)\s+([a-z_][a-z_0-9]*)", t)
    if m:
        return {"op": "get", "item": m.group(2)}
    m = re.search(r"\b(craft|make|build|create|combine)\s+(?:a |an |the )?([a-z_][a-z_0-9]*)", t)
    if m:
        return {"op": "craft", "item": m.group(2)}
    return {"op": "unknown", "item": None}


class TextCraftEnv(AgenticEnv):
    """Text crafting with a prerequisite recipe DAG (see module docstring)."""

    def __init__(self, seed: int = 0):
        self._seed = int(seed)
        self._counter = 0
        # episode state
        self._task: Optional[Task] = None
        self._target: str = ""
        self._recipes: Dict[str, List[str]] = {}      # item -> [ingredients]
        self._base: set = set()                       # gettable raw items
        self._subgoals: List[str] = []                # non-base items (predicate items)
        self._inv: Dict[str, int] = {}                # inventory counts
        self._got: set = set()                        # sub-goal items ever obtained
        self._done = False

    # ------------------------------------------------------------------ recipes
    def _level_for(self, difficulty: float) -> int:
        d = max(0.0, min(1.0, float(difficulty)))
        return 1 + int(round(d * 4))                  # depth 1..5

    def _gen_recipe_dag(self, rng: random.Random, depth: int) -> Dict[str, Any]:
        """Build a prerequisite DAG of the given depth rooted at a target item.

        Level `depth` = target (craftable). Level 0 = base (gettable). Each craftable
        item needs 2-3 ingredients drawn from the level below.
        """
        crafts = rng.sample(_CRAFT_POOL, k=min(len(_CRAFT_POOL), 1 + 2 * depth))
        bases = rng.sample(_BASE_POOL, k=min(len(_BASE_POOL), 2 + depth))
        # assign craftable items to levels 1..depth (target alone at top)
        target = crafts[0]
        levels: Dict[int, List[str]] = {depth: [target]}
        rest = crafts[1:]
        for lv in range(depth - 1, 0, -1):
            n = max(1, len(rest) // max(1, lv))
            levels[lv], rest = rest[:n], rest[n:]
            if not levels[lv]:
                levels[lv] = [rng.choice(_CRAFT_POOL)]
        levels[0] = bases
        recipes: Dict[str, List[str]] = {}
        for lv in range(depth, 0, -1):
            below = levels[lv - 1] + (levels.get(lv - 2, []) if lv >= 2 else [])
            for item in levels[lv]:
                k = rng.randint(2, 3)
                ings = rng.sample(below, k=min(k, len(below))) if below else []
                if not ings:
                    ings = [rng.choice(bases)]
                recipes[item] = ings
        base = set(levels[0])
        return {"target": target, "recipes": recipes, "base_items": sorted(base)}

    def _subgoals_of(self, target: str, recipes: Dict[str, List[str]]) -> List[str]:
        """Sub-goals = the craftable items in the TARGET's dependency closure ONLY.

        Recipes outside the closure stay in the recipe book as DISTRACTORS (the agent must
        figure out which recipes matter) but are not graded -- otherwise an unreachable
        distractor would cap verify_final below 1.0 even for a perfect solve.
        Returned in a valid craft order (dependencies before dependents).
        """
        order: List[str] = []
        seen: set = set()

        def visit(item: str, depth: int = 0):
            if item in seen or item not in recipes or depth > 64:
                return
            seen.add(item)
            for g in recipes[item]:
                visit(g, depth + 1)
            order.append(item)            # post-order => ingredients precede dependents

        visit(target)
        return order

    # ------------------------------------------------------------------ contract
    def sample_new_task(self, rng: Any, difficulty: float) -> Task:
        r = rng if isinstance(rng, random.Random) else random.Random(self._seed + self._counter)
        depth = self._level_for(difficulty)
        dag = self._gen_recipe_dag(r, depth)
        self._counter += 1
        tid = f"textcraft_d{depth}_{self._counter}"
        subgoals = self._subgoals_of(dag["target"], dag["recipes"])
        return Task(
            task_id=tid,
            spec={"goal": f"craft {dag['target']}", "target": dag["target"],
                  "recipes": dag["recipes"], "base_items": dag["base_items"]},
            constraints={"target": dag["target"], "recipes": dag["recipes"],
                         "base_items": dag["base_items"], "subgoals": subgoals},
            info={"difficulty": round(float(difficulty), 3), "depth": depth,
                  "n_subgoals": len(subgoals)},
        )

    def list_tasks(self, split: str) -> List[Task]:
        diffs = [0.25, 0.5, 0.75] if split == "train" else [0.5]
        return [self.sample_new_task(random.Random(self._seed + i), d)
                for i, d in enumerate(diffs)]

    def predicate_library(self, task: Task) -> List[str]:
        return [f"have_{it}" for it in task.constraints.get("subgoals", [])]

    def is_well_posed(self, task: Task) -> bool:
        rec = task.constraints.get("recipes", {})
        base = set(task.constraints.get("base_items", []))
        target = task.constraints.get("target", "")
        if not target or not rec or target not in rec:
            return False
        # every ingredient is either base or itself craftable (DAG closed) -> solvable
        for item, ings in rec.items():
            for g in ings:
                if g not in base and g not in rec:
                    return False
        return len(rec) > 0

    def reset(self, task: Task) -> Observation:
        self._task = task
        self._target = task.constraints["target"]
        self._recipes = {k: list(v) for k, v in task.constraints["recipes"].items()}
        self._base = set(task.constraints.get("base_items", []))
        self._subgoals = list(task.constraints.get("subgoals", []))
        self._inv = {}
        self._got = set()
        self._done = False
        return self._make_obs()

    def _make_obs(self) -> Observation:
        lines = [f"GOAL: craft '{self._target}'.", ""]
        lines.append("RECIPES (craft X consumes its ingredients; they must be in your inventory):")
        for item, ings in self._recipes.items():
            lines.append(f"  {item}  <=  " + " + ".join(ings))
        lines.append("")
        lines.append("BASE MATERIALS you can `get` directly: " + ", ".join(sorted(self._base)))
        inv = ", ".join(f"{k}x{v}" for k, v in sorted(self._inv.items()) if v > 0) or "(empty)"
        lines.append(f"INVENTORY: {inv}")
        lines.append("")
        lines.append("Actions: `get <base_item>` | `craft <item>` | `done`")
        return Observation(
            text="\n".join(lines),
            available_actions=["get <base_item>", "craft <item>", "done"],
            state={"inventory": dict(self._inv), "target": self._target},
        )

    def step(self, action: Action) -> StepResult:
        if self._task is None:
            raise RuntimeError("step() before reset()")
        # The shared roles.py Executor fills action.parsed with a DeepPlanning-grammar dict
        # ({"verb","item_index",...}); it has no 'op', so we ALWAYS parse from raw with the
        # TextCraft grammar (the env owns its own action language).
        p = _parse(action.raw)
        op, item = p.get("op"), p.get("item")
        records: List[StageRecord] = []
        info: Dict[str, Any] = {}

        if op == "done":
            self._done = True
            return StepResult(obs=self._make_obs(), stage_records=[], done=True,
                              terminal_reward=self.verify_final(),
                              info={"reached_done": True})

        if op == "get":
            if item in self._base:
                self._inv[item] = self._inv.get(item, 0) + 1
            else:
                info["failure_type"] = "constraint_violation"
                info["msg"] = f"`get {item}` invalid: only base materials are gettable"
                records.append(StageRecord(predicate_id=f"get_{item}", before="unsat",
                                           after="unsat", failure_type="constraint_violation",
                                           char_span=(0, 0), info={"item": str(item)}))
        elif op == "craft":
            rec = self._recipes.get(item)
            pid = f"have_{item}"
            if rec is None:
                info["failure_type"] = "format"
                records.append(StageRecord(predicate_id=pid, before="unsat", after="unsat",
                                           failure_type="format", char_span=(0, 0),
                                           info={"msg": f"no recipe for {item!r}"}))
            else:
                missing = [g for g in rec if self._inv.get(g, 0) < rec.count(g)]
                before = "sat" if item in self._got else "unsat"
                if missing:
                    # THE long-horizon failure: crafting before prerequisites are ready.
                    info["failure_type"] = "missing"
                    info["needs_replan"] = True
                    info["missing"] = missing
                    records.append(StageRecord(predicate_id=pid, before=before, after=before,
                                               failure_type="missing", char_span=(0, 0),
                                               info={"missing": missing}))
                else:
                    for g in rec:                       # consume ingredients
                        self._inv[g] = self._inv.get(g, 0) - 1
                    self._inv[item] = self._inv.get(item, 0) + 1
                    self._got.add(item)
                    records.append(StageRecord(predicate_id=pid, before=before, after="sat",
                                               failure_type="none", char_span=(0, 0),
                                               info={}))
        else:
            info["failure_type"] = "format"
            info["msg"] = "unparseable action; use `get <item>` / `craft <item>` / `done`"

        done = self._inv.get(self._target, 0) > 0       # auto-terminate on success
        if done:
            self._done = True
        tr = self.verify_final() if done else None
        return StepResult(obs=self._make_obs(), stage_records=records, done=done,
                          terminal_reward=tr, info=info)

    def verify_final(self) -> float:
        if self._task is None:
            return 0.0
        if self._inv.get(self._target, 0) > 0:
            return 1.0
        sg = self._subgoals or [self._target]
        return sum(1 for it in sg if it in self._got) / float(len(sg))

    def needs_replan(self, last: StepResult) -> bool:
        return bool(last.info.get("needs_replan", False)) if last else False
