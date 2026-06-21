"""
Task and Prompt Noise Injection Transformer.

This script augments a *leveled* game task dataset and its corresponding
prompt strings by injecting "noise" items—additional craftable/equipable
items relevant to the target monster but not strictly required. The goal
is to increase task complexity, diversify strategies, and test agent robustness.

It performs three main operations:

1. **Noise Item Injection**:
   - Loads a catalog of candidate noise items from the game database.
   - Profiles the target monster’s elemental weaknesses/attacks.
   - Picks helpful, diverse items (avoiding monster drops and trivial items).
   - Fills any remaining quota with random valid craftables.
   - Updates each task’s `task_info` with noise items and their dependency data
     (crafting ingredients, resources, locations).

2. **Task Dataset Augmentation**:
   - Iterates through all tasks in the input leveled dataset.
   - Injects noise items into each applicable task.
   - Writes the resulting noise-augmented dataset to a new JSON file.

3. **Prompt String Patching**:
   - Matches each transformed task to its corresponding prompt string.
   - Replaces the in-text `task_info` structure with the updated version.
   - Writes a new prompt dataset aligned with the noise-augmented tasks.

Default input/output (same spirit as level_transformer.py):
    --tasks_in       datasets/dataset_tasks_leveling.json
    --prompts_in     datasets/dataset_prompts_leveling.json
    --tasks_out      datasets/dataset_tasks_noise_leveling.json
    --prompts_out    datasets/dataset_prompts_noise_leveling.json
    --noise_catalog  Virtual_Environment/Data/noise_items.json
    --noise_n        10

Example:
    python noise_transformer.py \
        --tasks_in datasets/dataset_tasks_leveling.json \
        --prompts_in datasets/dataset_prompts_leveling.json \
        --tasks_out datasets/dataset_tasks_noise_leveling.json \
        --prompts_out datasets/dataset_prompts_noise_leveling.json \
        --noise_catalog Virtual_Environment/Data/noise_items.json \
        --noise_n 12

"""

import argparse
import json
import re
from pathlib import Path
from random import sample
import random
import itertools

# ─────────────────────────────────────────────────────────
# Imports from your existing module(s)
# ─────────────────────────────────────────────────────────
from crafting_tree import (
    build_crafting_tree,
    items_by_code,
    MONSTERS_DATA,
    ITEMS_DATA,
)

# (If these helpers are in crafting_tree; if not, import where they live)
# from crafting_tree import drops_by_item, resources_by_item, locations_by_monster, locations_by_resource

# ─────────────────────────────────────────────────────────
# 0.  Safe build_crafting_tree wrapper (as you already had)
# ─────────────────────────────────────────────────────────
import crafting_tree as _ct
_original_bct = _ct.build_crafting_tree

def _prune_unknown(node):
    if not isinstance(node, dict):
        return
    if "craft" in node:
        keep = []
        for ing in node["craft"].get("ingredients", []):
            code = ing.get("code")
            if code is None or code in items_by_code or "craft" in ing:
                _prune_unknown(ing)
                keep.append(ing)
        node["craft"]["ingredients"] = keep

def build_crafting_tree_safe(item_dict):
    tree = _original_bct(item_dict)
    _prune_unknown(tree)
    return tree

_ct.build_crafting_tree = build_crafting_tree_safe

# ─────────────────────────────────────────────────────────
# 1.  Lookups
# ─────────────────────────────────────────────────────────
monsters_by_code = {m["code"]: m for m in MONSTERS_DATA}

# ─────────────────────────────────────────────────────────
# 2.  Utility load
# ─────────────────────────────────────────────────────────
def load_json(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)

# ─────────────────────────────────────────────────────────
# 3.  update_variant_with_valid_items (your existing version)
# ─────────────────────────────────────────────────────────
def update_variant_with_valid_items(
        variant,
        valid_items=None,
        noise_items=None,
        *,
        num_random_monsters=0,
        num_random_craftable=0,
        num_random_locations=0):

    task_info      = variant.setdefault("task_info", {})
    monsters_list  = task_info.setdefault("Monsters", [])
    craft_list     = task_info.setdefault("Craftable items", [])
    resources_list = task_info.setdefault("Resources", [])
    locations_list = task_info.setdefault("Locations", [])
    items_stats    = task_info.setdefault("Items stats", [])

    monster_codes   = {m.get("code") for m in monsters_list if "code" in m}
    craft_codes     = {i.get("code") for i in craft_list   if "code" in i}
    resource_codes  = {r.get("code") for r in resources_list if "code" in r}
    location_keys   = {(l.get("name"), l.get("skin"), l.get("x"), l.get("y"),
                        (l.get("content") or {}).get("code", "")) for l in locations_list}

    def _traverse_tree(node):
        if "craft" in node:
            code = node.get("code")
            if code and code not in craft_codes and code in items_by_code:
                craft_list.append(items_by_code[code])
                craft_codes.add(code)
            for ing in node["craft"].get("ingredients", []):
                _traverse_tree(ing)
        else:
            code = node.get("code")
            if code and code not in resource_codes and code in items_by_code:
                resources_list.append(items_by_code[code])
                resource_codes.add(code)

            for mon in node.get("monsters", []):
                m_code = mon.get("code")
                if m_code and m_code not in monster_codes and m_code in monsters_by_code:
                    monsters_list.append(monsters_by_code[m_code])
                    monster_codes.add(m_code)

                for loc in mon.get("locations", []):
                    key = (loc.get("name"), loc.get("skin"), loc.get("x"), loc.get("y"),
                           (loc.get("content") or {}).get("code", ""))
                    if key not in location_keys:
                        locations_list.append(loc)
                        location_keys.add(key)

            for res in node.get("resources", []):
                r_code = res.get("resource_code")
                if r_code and r_code not in resource_codes and r_code in items_by_code:
                    resources_list.append(items_by_code[r_code])
                    resource_codes.add(r_code)
                for loc in res.get("locations", []):
                    key = (loc.get("name"), loc.get("skin"), loc.get("x"), loc.get("y"),
                           (loc.get("content") or {}).get("code", ""))
                    if key not in location_keys:
                        locations_list.append(loc)
                        location_keys.add(key)

    if valid_items:
        for slot, code in valid_items.items():
            if not code:
                continue
            if code not in items_by_code:
                print(f"Warning: Item with code '{code}' not found in ITEMS_DATA.")
                continue
            item = items_by_code[code]
            items_stats.append({k: v for k, v in item.items() if k != "craft"})

    if noise_items:
        variant["noise_items"] = list(noise_items)
        if "equiped items" in variant:
            new_variant = {}
            for k, v in variant.items():
                new_variant[k] = v
                if k == "equiped items":
                    new_variant["noise_items"] = variant["noise_items"]
            for k in variant:
                if k not in new_variant:
                    new_variant[k] = variant[k]
            variant.clear()
            variant.update(new_variant)

        for slot, code in noise_items:
            if not code or code not in items_by_code:
                print(f"Warning: Noise item '{code}' not found in ITEMS_DATA.")
                continue
            if code not in craft_codes:
                craft_list.append(items_by_code[code])
                craft_codes.add(code)
            _traverse_tree(build_crafting_tree(items_by_code[code]))

    if num_random_monsters:
        pool = [m for c, m in monsters_by_code.items() if c not in monster_codes]
        for m in random.sample(pool, min(num_random_monsters, len(pool))):
            monsters_list.append(m); monster_codes.add(m["code"])

    if num_random_craftable:
        pool = [i for c, i in items_by_code.items() if i.get("craft") and c not in craft_codes]
        for i in random.sample(pool, min(num_random_craftable, len(pool))):
            craft_list.append(i); craft_codes.add(i["code"])

    if num_random_locations:
        try:
            maps = json.loads(Path("app/Data/maps.json").read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Error loading maps.json: {exc}")
            maps = []
        available = []
        for loc in maps:
            key = (loc.get("name"), loc.get("skin"), loc.get("x"), loc.get("y"),
                   (loc.get("content") or {}).get("code", ""))
            if key not in location_keys:
                available.append((key, loc))
        for key, loc in random.sample(available, min(num_random_locations, len(available))):
            locations_list.append(loc); location_keys.add(key)

    return variant

# ─────────────────────────────────────────────────────────
# 4.  Noise injection helper (add_noise_items)
# ─────────────────────────────────────────────────────────
def add_noise_items(task_blob: dict,
                    all_items_catalog: list,
                    n: int) -> dict:
    """Inject up to *n* helpful 'noise' items against the task’s target monster.
    Any shortfall is backfilled with random valid items.
    """

    ELEMENTS = ("fire", "earth", "water", "air")

    def _monster_from_task(tb):
        tgt_name = tb.get("monster_name")
        for m in tb["task_info"].get("Monsters", []):
            if m.get("name") == tgt_name or m.get("code") == tgt_name:
                return m
        return tb["task_info"]["Monsters"][0]

    def _weak_elements(mon):
        res = {el: mon.get(f"res_{el}", 0) for el in ELEMENTS}
        lo = min(res.values())
        return {el for el, v in res.items() if v == lo}

    def _strong_attack_elements(mon):
        atk = {el: mon.get(f"attack_{el}", 0) for el in ELEMENTS}
        hi = max(atk.values())
        return {el for el, v in atk.items() if v == hi and v > 0}

    monster = _monster_from_task(task_blob)
    weak_elems   = _weak_elements(monster)
    strong_elems = _strong_attack_elements(monster)

    craftables_by_code = {
        c["code"]: c.get("level", 0)
        for c in task_blob["task_info"]["Craftable items"]
        if isinstance(c, dict) and "code" in c
    }
    missing_lvls = [
        craftables_by_code[c]
        for c in task_blob.get("missing_items", {}).values()
        if c in craftables_by_code
    ]
    if not missing_lvls:
        raise ValueError("No level info for missing items (nothing to base filter on)")
    max_missing_lvl = max(missing_lvls)

    monster_drop_codes = {
        drop["code"]
        for m in task_blob["task_info"].get("Monsters", [])
        for drop in m.get("drops", [])
        if "code" in drop
    }

    def valid_noise(itm):
        if not isinstance(itm, dict):
            return False
        if itm.get("type") in {"resource", "consumable"}:
            return False
        if itm.get("craft") is None:
            return False
        if itm.get("code") in monster_drop_codes:
            return False
        if itm.get("level", 0) < max_missing_lvl - 10:
            return False
        return True

    def relevance_score(itm):
        sc = 0
        for eff in itm.get("effects", []):
            name, val = eff.get("name", ""), eff.get("value", 0)
            if name.startswith(("attack_", "dmg_")):
                el = name.split("_")[1]
                if el in weak_elems:
                    sc += val or 10
            elif name.startswith("res_"):
                el = name.split("_")[1]
                if el in strong_elems:
                    sc += val or 5
        return sc

    candidates = [i for i in all_items_catalog if valid_noise(i)]
    if not candidates:
        raise ValueError("No valid candidates in noise catalogue")

    ranked = sorted(candidates, key=relevance_score, reverse=True)
    good   = [i for i in ranked if relevance_score(i) > 0]
    random_pool = [i for i in ranked if i not in good]

    chosen = []
    used_types = set()

    def _take_from(pool):
        nonlocal chosen, used_types
        for itm in pool:
            if len(chosen) >= n:
                break
            t = itm.get("type")
            if t in used_types:
                continue
            chosen.append(itm)
            used_types.add(t)

    _take_from(good)
    if len(chosen) < n:
        sample_size = min(n - len(chosen), len(random_pool))
        _take_from(random.sample(random_pool, sample_size))

    if len(chosen) < n:
        missing = n - len(chosen)
        extras = random.sample([i for i in candidates if i not in chosen],
                               min(missing, len(candidates) - len(chosen)))
        chosen.extend(extras)

    def infer_slot(item_dict):
        if item_dict.get("slot"):
            return item_dict["slot"]
        return f"{item_dict['type']}_slot"

    noise_pairs = []
    for itm in chosen:
        code = itm.get("code")
        if not code:
            continue
        if code not in items_by_code:
            items_by_code[code] = itm
        noise_pairs.append((infer_slot(itm), code))

    update_variant_with_valid_items(task_blob, noise_items=noise_pairs)

    min_lvl = task_blob.get("min_level")
    if min_lvl is not None:
        craft_list = task_blob["task_info"]["Craftable items"]
        for _, code in noise_pairs:
            for entry in craft_list:
                if entry.get("code") == code:
                    entry["level"] = min_lvl

    return task_blob

# ─────────────────────────────────────────────────────────
# 5.  Process all tasks then patch prompt strings
# ─────────────────────────────────────────────────────────
def process_all_tasks(tasks_json: dict, catalog: list, n: int):
    for key, task_list in tasks_json.items():
        for i, task in enumerate(task_list):
            if not isinstance(task, dict):
                continue
            if "monster_name" not in task or "task_info" not in task:
                continue
            try:
                add_noise_items(task, catalog, n)
            except Exception as e:
                mname = task.get("monster_name", "N/A")
                tdiff = task.get("total_difficulty", "N/A")
                print(f"Skipping task [key={key}] (monster={mname}, diff={tdiff}) due to error: {e}")
    return tasks_json

# ─────────────────────────────────────────────────────────
# 6.  STRING PATCHING (new for noise pass)
# ─────────────────────────────────────────────────────────
MONSTER_LINE_RE = re.compile(r"\n\{'Monsters[^\n]*")  # existing anchor

def patch_prompt_with_task_info(raw_text: str, task_info: dict) -> str:
    """
    Replace the line beginning with {'Monsters ... } with repr(task_info).
    If anchor not found, returns original text.
    """
    if "\n{'Monsters" not in raw_text:
        return raw_text
    return MONSTER_LINE_RE.sub("\n" + repr(task_info), raw_text, count=1)

def transform_strings_after_noise(tasks_json: dict,
                                  strings_in_path: Path,
                                  strings_out_path: Path):
    strings_json = json.loads(strings_in_path.read_text(encoding="utf-8"))
    for diff_key, task_list in tasks_json.items():
        if diff_key not in strings_json:
            continue
        updated_bucket = []
        original_bucket = strings_json[diff_key]
        for task, txt in itertools.zip_longest(task_list, original_bucket, fillvalue=None):
            if task is None or txt is None:
                updated_bucket.append(txt or "")
                continue
            updated_bucket.append(patch_prompt_with_task_info(txt, task.get("task_info", {})))
        strings_json[diff_key] = updated_bucket

    strings_out_path.write_text(json.dumps(strings_json, indent=2, ensure_ascii=False),
                                encoding="utf-8")
    print(f"Updated noise prompt-strings → {strings_out_path}")

# ─────────────────────────────────────────────────────────
# 7.  CLI entry-point (argparse, like File 1)
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks_in",      default="datasets/dataset_tasks_leveling.json")
    parser.add_argument("--prompts_in",    default="datasets/dataset_prompts_leveling.json")
    parser.add_argument("--tasks_out",     default="datasets/dataset_tasks_noise_leveling.json")
    parser.add_argument("--prompts_out",   default="datasets/dataset_prompts_noise_leveling.json")
    parser.add_argument("--noise_catalog", default="Virtual_Environment/Data/noise_items.json")
    parser.add_argument("--noise_n",       type=int, default=10,
                        help="How many noise items to try to inject per task")
    args = parser.parse_args()

    TASKS_IN    = Path(args.tasks_in)
    PROMPTS_IN  = Path(args.prompts_in)
    TASKS_OUT   = Path(args.tasks_out)
    PROMPTS_OUT = Path(args.prompts_out)
    CATALOG     = Path(args.noise_catalog)
    NOISE_N     = int(args.noise_n)

    catalog = load_json(str(CATALOG))
    for itm in catalog:
        code = itm.get("code")
        if code and code not in items_by_code:
            items_by_code[code] = itm

    data = load_json(str(TASKS_IN))
    augmented = process_all_tasks(data, catalog, n=NOISE_N)
    TASKS_OUT.parent.mkdir(parents=True, exist_ok=True)
    TASKS_OUT.write_text(json.dumps(augmented, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote tasks with noise → {TASKS_OUT}")

    transform_strings_after_noise(augmented, PROMPTS_IN, PROMPTS_OUT)
