"""
Task and Prompt Leveling Transformer.

This script processes a dataset of game tasks and their associated prompt
strings to introduce skill leveling mechanics for resource gathering
(e.g., mining, woodcutting) and crafting. It performs two main operations:

1. **Task Transformation**:
   - Loads item, resource, and location data from the game database.
   - Determines the required profession levels to craft the items in each task.
   - Adds intermediate resources and locations to the task based on the
     player's starting skill level (set to level 1) and the target skill level.
   - Updates the task's `task_info` section with the expanded resource and
     location lists.

2. **Prompt String Transformation**:
   - Matches each transformed task to its corresponding prompt string.
   - Updates in-text game data such as character stats and monsters.
   - Resets gathering skill levels to 1 and adds a paragraph explaining the
     XP/leveling system for gathering and crafting professions.

The transformed task and prompt datasets are saved to new JSON files,
as specified by command-line arguments.

Default input/output:
    --tasks        datasets/dataset_tasks.json
    --out          datasets/dataset_tasks_leveling.json
    --strings      datasets/dataset_prompts.json
    --strings_out  datasets/dataset_prompts_leveling.json

Example:
    python level_transformer.py \
        --tasks datasets/dataset_tasks.json \
        --out datasets/dataset_tasks_leveling.json \
        --strings datasets/dataset_prompts.json \
        --strings_out datasets/dataset_prompts_leveling.json
"""

from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────
# 1.  Locate input / output (falls back to hard-coded paths)
# ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--tasks",   default="datasets/dataset_tasks.json")
parser.add_argument("--out",     default="datasets/dataset_tasks_leveling.json")
parser.add_argument("--strings", default="datasets/dataset_prompts.json")
parser.add_argument("--strings_out",
                    default="datasets/dataset_prompts_leveling.json")
args = parser.parse_args()

TASKS_IN   = Path(args.tasks)
TASKS_OUT  = Path(args.out)
STR_IN     = Path(args.strings)
STR_OUT    = Path(args.strings_out)

# ────────────────────────────────────────────────────────────────────────
# 2.  Load databases used by the *task* transformer (unchanged)
# ────────────────────────────────────────────────────────────────────────
BASE = Path("app/Data")                      # adjust if your paths differ
ITEMS_DATA     = json.loads(Path(BASE / "items.json").read_text(encoding="utf-8"))
RESOURCES_DATA = json.loads(Path(BASE / "resources.json").read_text(encoding="utf-8"))
LOCATIONS_DATA = json.loads(Path(BASE / "maps.json").read_text(encoding="utf-8"))

ITEMS_BY_CODE  = {i["code"]: i for i in ITEMS_DATA}
RES_NODE_BY_CODE = {r["code"]: r for r in RESOURCES_DATA}

ORES_ITEMS = sorted(
    (i for i in ITEMS_DATA if i.get("subtype") == "mining"),
    key=lambda x: x["level"],
)
LOGS_ITEMS = sorted(
    (i for i in ITEMS_DATA if i.get("subtype") == "woodcutting"),
    key=lambda x: x["level"],
)

# ────────────────────────────────────────────────────────────────────────
# 3.  Helper look-up tables used by the *task* transformer (unchanged)
# ────────────────────────────────────────────────────────────────────────
ITEM_SOURCES: dict[str, list[str]] = defaultdict(list)
for node in RESOURCES_DATA:
    for drop in node.get("drops", []):
        ITEM_SOURCES[drop["code"]].append(node["code"])

LOC_BY_NODE: dict[str, list[dict]] = defaultdict(list)
for loc in LOCATIONS_DATA:
    content = loc.get("content") or {}
    if content.get("type") == "resource":
        LOC_BY_NODE[content["code"]].append(loc)

# ────────────────────────────────────────────────────────────────────────
# 4.  Crafting-tree helpers (unchanged)
# ────────────────────────────────────────────────────────────────────────
def walk_ingredients(code: str, seen: set[str] | None = None):
    seen = seen or set()
    if code in seen or code not in ITEMS_BY_CODE:
        return
    seen.add(code)
    yield code
    craft = ITEMS_BY_CODE[code].get("craft")
    if craft:
        for ing in craft["items"]:
            yield from walk_ingredients(ing["code"], seen)


def required_caps(product_codes: list[str]) -> dict[str, int]:
    caps = {"mining": 1, "woodcutting": 1}
    for code in product_codes:
        for ing in walk_ingredients(code):
            itm = ITEMS_BY_CODE[ing]
            skill = itm.get("subtype")                # raw ore / log
            if skill in caps:
                caps[skill] = max(caps[skill], itm["level"])
            craft = itm.get("craft")
            if craft and craft["skill"] in caps:
                caps[craft["skill"]] = max(caps[craft["skill"]], craft["level"])
    return caps


def build_ladder(skill: str, target_lvl: int) -> list[dict]:
    pool = ORES_ITEMS if skill == "mining" else LOGS_ITEMS
    return [res for res in pool if res["level"] <= target_lvl]

# ────────────────────────────────────────────────────────────────────────
# 5.  Main TASK transformer (unchanged)
# ────────────────────────────────────────────────────────────────────────
def process_task(task: dict):
    char = task.setdefault("character", {})
    for sk in ("mining", "woodcutting"):
        char[f"{sk}_level"]  = 1
        char[f"{sk}_xp"]     = 0
        char[f"{sk}_max_xp"] = 150

    ti = task.setdefault("task_info", {})
    product_codes = [itm["code"] for itm in ti.get("Craftable items", [])]
    caps = required_caps(product_codes)

    resources_section: dict[str, dict] = {r["code"]: r for r in ti.get("Resources", [])}
    locations_list:  list[dict]        = list(ti.get("Locations", []))

    for skill, target_lvl in caps.items():
        if target_lvl == 1:
            continue
        for res_item in build_ladder(skill, target_lvl):
            code = res_item["code"]
            entry = res_item.copy()
            entry["sources"] = ITEM_SOURCES.get(code, [])
            resources_section[code] = entry

            for node_code in entry["sources"]:
                for loc in LOC_BY_NODE.get(node_code, []):
                    if loc not in locations_list:
                        locations_list.append(loc)

    ti["Resources"] = sorted(resources_section.values(), key=lambda x: x["level"])
    ti["Locations"] = locations_list
    task["task_info"] = ti
    return task


def transform_tasks(in_path: Path, out_path: Path) -> dict:
    tasks_json = json.loads(in_path.read_text(encoding="utf-8"))

    for diff_key, task_list in tasks_json.items():
        tasks_json[diff_key] = [process_task(t) for t in task_list]

    out_path.write_text(json.dumps(tasks_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated tasks → {out_path}")
    return tasks_json   # hand this to the string-pass next

# ────────────────────────────────────────────────────────────────────────
# 6.  TEXT-STRING transformer (new)
# ────────────────────────────────────────────────────────────────────────
XP_PARAGRAPH = (
    "To gather resources your corresponding profession should be apropriate level (mining, woodcutting, etc). You gain experience in each profession (gathering or crafting) by performing relevant actions. Each time you perform an action in a profession like mining or woodcutting, you gain 50 XP for that profession unless your skill level is more than 10 levels above the action’s required level. To reach the next level, you need 150 XP for level 1, with each subsequent level requiring 10 XP more than the previous."
)

MONSTER_LINE_RE = re.compile(r"\n\{'Monsters[^\n]*")                # whole monsters line
MIN_LVL_RE      = re.compile(r"'mining_level':\s*\d+")
WOOD_LVL_RE     = re.compile(r"'woodcutting_level':\s*\d+")

def patch_string(text: str, task_info: dict) -> str | None:
    """
    Try to patch a single text block. If the expected anchors are missing,
    return None so the caller can fall back to the unmodified original.
    """
    if "\n{'Monsters" not in text or "\nCharacter Stats:\n" not in text:
        return None

    # 1) replace monsters-line with task_info
    text = MONSTER_LINE_RE.sub("\n" + repr(task_info), text, count=1)

    # 2) bump both gather skills down to 1
    text = MIN_LVL_RE.sub("'mining_level': 1", text, count=1)
    text = WOOD_LVL_RE.sub("'woodcutting_level': 1", text, count=1)

    # 3) inject XP explanation after the given sentence
    anchor = "Your crafting level should not be less than required."
    if anchor in text:
        text = text.replace(anchor, f"{anchor}\n{XP_PARAGRAPH}", 1)

    return text


def transform_strings(tasks_json: dict, str_in: Path, str_out: Path):
    """Match every task to its corresponding text entry and patch it."""
    strings_json = json.loads(str_in.read_text(encoding="utf-8"))

    for diff_key, task_list in tasks_json.items():
        if diff_key not in strings_json:        # no text for this bucket → skip
            continue

        new_bucket: list[str] = []
        for task, raw_text in zip(task_list, strings_json[diff_key]):
            patched = patch_string(raw_text, task["task_info"])
            new_bucket.append(patched if patched is not None else raw_text)

        strings_json[diff_key] = new_bucket

    str_out.write_text(json.dumps(strings_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated task-strings → {str_out}")

# ────────────────────────────────────────────────────────────────────────
# 7.  CLI entry-point
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    transformed_tasks = transform_tasks(TASKS_IN, TASKS_OUT)
    transform_strings(transformed_tasks, STR_IN, STR_OUT)
