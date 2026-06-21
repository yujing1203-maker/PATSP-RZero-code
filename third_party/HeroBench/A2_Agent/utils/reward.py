import os
import re
import json
import threading
from typing import Tuple, Optional
import multiprocessing
import traceback
from collections import deque
from typing import Dict, List, Any
from collections import defaultdict
from pathlib import Path

import utils.crafting_tree as ct       


# ---------------------------------------------------------------------------
#  Helper: consistent monster key normalization
# ---------------------------------------------------------------------------
def norm_monster(name: str) -> str:
    """Normalize monster names for consistent keys."""
    return re.sub(r'\s+', ' ', name.strip().lower())

def _add(breakdown: dict, key: str) -> int:
    """Increment a breakdown entry and return +1 for total."""
    breakdown[key] = breakdown.get(key, 0) + 1
    return 1

# ---------------------------------------------------------------------------
#  Scoring weights
# ---------------------------------------------------------------------------
WEIGHTS = {
    'target_win':          1,
    'craft_missing':       1,
    'equip_missing':       1,
    'obtain_missing':      1,
    'other_monster':       1,
    'gather_resource':     1,
    'craft_intermediate':  1,
    'buy_resource':        1,
}

# ---------------------------------------------------------------------------
#  Regex patterns
# ---------------------------------------------------------------------------
R_FIGHT = re.compile(
    r"\b(?P<result>win|lost) (?:his|their|your)? fight against (?P<monster>[\w\s&']+)",
    re.I
)
R_CRAFT  = re.compile(r"crafts (\w+) x", re.I)
R_EQUIP  = re.compile(r"equipped item (\w+)", re.I)
R_GATHER = re.compile(
    r"gathered resources\s+([\w_]+(?:\s*,\s*[\w_]+)*)(?=\s+with|$)",
    re.I
)
R_BUY    = re.compile(r"(?:bought|purchased)\s+(\w+)", re.I)

# ---------------------------------------------------------------------------
#  Helper: gatherable leaf resources in a crafting tree
# ---------------------------------------------------------------------------
def gatherable_resources(tree: dict) -> set[str]:
    if tree.get('basic'):
        return {tree['code']}
    need = set()
    for sub in tree['craft']['ingredients']:
        need |= gatherable_resources(sub)
    return need

# ---------------------------------------------------------------------------
#  Public API: route tasks to correct scorer
# ---------------------------------------------------------------------------
def compute_episode_reward(task: dict, logs: list[dict]) -> tuple[int, dict[str,int]]:
    is_kill = bool(task.get('monster_name'))
    return _reward_kill(task, logs) if is_kill else _reward_craft(task, logs)

# ---------------------------------------------------------------------------
#  Kill-and-Craft tasks
# ---------------------------------------------------------------------------
def _reward_kill(task: dict, logs: list[dict]) -> tuple[int, dict[str,int]]:
    target   = task.get('monster_name','').lower()
    missing  = set(task['missing_items'].values())

    # Required basic resources
    required_resources = set()
    for code in missing:
        tree = ct.build_crafting_tree(ct.items_by_code[code])
        required_resources |= gatherable_resources(tree)

    # Intermediate codes
    intermediate = set()
    def _collect(tree: dict):
        if 'craft' in tree:
            code = tree['code']
            if code not in missing:
                intermediate.add(code)
            for ing in tree['craft']['ingredients']:
                _collect(ing)
    for code in missing:
        _collect(ct.build_crafting_tree(ct.items_by_code[code]))

    # Drop sources: code -> monsters
    drop_sources = defaultdict(set)
    for code in missing:
        for d in ct.drops_by_item.get(code, []):
            drop_sources[code].add(d['name'])

    got_target = False
    beaten     = set()
    crafted_m  = set()
    crafted_i  = set()
    equipped   = set()
    obtained   = set()
    gathered   = set()
    total      = 0
    breakdown  = defaultdict(int)

    for entry in logs:
        line = entry['log']

        # – Fight –
        if m := R_FIGHT.search(line):
            res = m.group('result').lower()
            mon_raw = m.group('monster')
            mon = norm_monster(mon_raw)

            if res in ('win','wins','won'):
                if mon == target:
                    if not got_target:
                        total += WEIGHTS['target_win']
                        breakdown['target_win'] += WEIGHTS['target_win']
                        got_target = True
                else:
                    if mon not in beaten:
                        total += WEIGHTS['other_monster']
                        breakdown[f'other_monster:{mon}'] += WEIGHTS['other_monster']
                        beaten.add(mon)

                # Drops
                for code, mons in drop_sources.items():
                    normalized = {norm_monster(x) for x in mons}
                    if code not in obtained and mon in normalized:
                        total += WEIGHTS['obtain_missing']
                        breakdown[f'obtain_missing:{code}'] += WEIGHTS['obtain_missing']
                        obtained.add(code)

        # – Craft –
        if c := R_CRAFT.search(line):
            code = c.group(1).lower()
            if code in missing and code not in crafted_m:
                total += WEIGHTS['craft_missing']
                breakdown[f'craft_missing:{code}'] += WEIGHTS['craft_missing']
                crafted_m.add(code)
            elif code in intermediate and code not in crafted_i:
                total += WEIGHTS['craft_intermediate']
                breakdown[f'craft_intermediate:{code}'] += WEIGHTS['craft_intermediate']
                crafted_i.add(code)

        # – Equip –
        if e := R_EQUIP.search(line):
            code = e.group(1).lower()
            if code in missing and code not in equipped:
                total += WEIGHTS['equip_missing']
                breakdown[f'equip_missing:{code}'] += WEIGHTS['equip_missing']
                equipped.add(code)

        # – Gather –
        if g := R_GATHER.search(line):
            for r in [r.strip().lower() for r in g.group(1).split(',')]:
                if r in required_resources and r not in gathered:
                    total += WEIGHTS['gather_resource']
                    breakdown[f'gather_resource:{r}'] += WEIGHTS['gather_resource']
                    gathered.add(r)

    return total, dict(breakdown), got_target

# ---------------------------------------------------------------------------
#  Pure-Craft tasks
# ---------------------------------------------------------------------------
def _reward_craft(task: dict, logs: list[dict]) -> tuple[int, dict[str,int]]:
    tree   = task['crafting_tree']
    target = tree['code'].lower()

    intermediate = set()
    basic        = set()
    def _walk(t: dict):
        c = t['code'].lower()
        if 'craft' in t:
            if c != target:
                intermediate.add(c)
            for ing in t['craft']['ingredients']:
                _walk(ing)
        else:
            basic.add(c)
    _walk(tree)

    crafted_t = False
    crafted_i = set()
    gathered  = set()
    bought    = set()
    total     = 0
    breakdown = defaultdict(int)

    for entry in logs:
        line = entry['log']

        if c := R_CRAFT.search(line):
            code = c.group(1).lower()
            if code == target and not crafted_t:
                total += WEIGHTS['target_win']
                breakdown['target_win'] += WEIGHTS['target_win']
                crafted_t = True
            elif code in intermediate and code not in crafted_i:
                total += WEIGHTS['craft_intermediate']
                breakdown[f'craft_intermediate:{code}'] += WEIGHTS['craft_intermediate']
                crafted_i.add(code)

        if g := R_GATHER.search(line):
            for r in [r.strip().lower() for r in g.group(1).split(',')]:
                if r in basic and r not in gathered:
                    total += WEIGHTS['gather_resource']
                    breakdown[f'gather_resource:{r}'] += WEIGHTS['gather_resource']
                    gathered.add(r)

        if b := R_BUY.search(line):
            code = b.group(1).lower()
            if code in basic and code not in bought:
                total += WEIGHTS['buy_resource']
                breakdown[f'buy_resource:{code}'] += WEIGHTS['buy_resource']
                bought.add(code)

    return total, dict(breakdown), crafted_t

# ---------------------------------------------------------------------------
#  Ideal (perfect-play) scorer
# ---------------------------------------------------------------------------
def compute_ideal_episode_reward(task: dict) -> tuple[int, dict[str,int]]:
    breakdown = defaultdict(int)
    total     = 0

    # Kill-and-craft
    if task.get('monster_name'):
        boss    = task['monster_name'].lower()
        total  += _add(breakdown, 'target_win')

        missing      = set(task['missing_items'].values())
        intermediate = set()
        basic        = set()

        def _walk2(t: dict):
            c = t['code'].lower()
            if c == 'wooden_stick':
                return
            if 'craft' in t:
                if c not in missing:
                    intermediate.add(c)
                for ing in t['craft']['ingredients']:
                    _walk2(ing)
            else:
                basic.add(c)

        for code in missing:
            _walk2(ct.build_crafting_tree(ct.items_by_code[code]))

        # Decide gatherable vs mob-drop by actual drop mapping
        gatherable = {c for c in basic if not ct.drops_by_item.get(c)}
        mob_drop   = basic - gatherable

        # Map monsters -> codes they drop
        monster_to_codes = defaultdict(set)
        for code in mob_drop:
            for d in ct.drops_by_item.get(code, []):
                monster_to_codes[norm_monster(d['name'])].add(code)

        # Include every monster that can drop a needed item
        for mon in monster_to_codes:
            if mon != boss:
                total += _add(breakdown, f'other_monster:{mon}')

        # Missing equipment
        for code in missing:
            if ct.items_by_code[code].get('craft'):
                total += _add(breakdown, f'craft_missing:{code}')
            else:
                total += _add(breakdown, f'obtain_missing:{code}')
            total += _add(breakdown, f'equip_missing:{code}')

        # Intermediates
        for code in intermediate:
            total += _add(breakdown, f'craft_intermediate:{code}')

        # Gatherable basics
        for code in gatherable:
            total += _add(breakdown, f'gather_resource:{code}')

        return total, dict(breakdown)

    # Pure-craft
    tree   = task['crafting_tree']
    target = tree['code'].lower()
    total += _add(breakdown, 'target_win')

    intermediate = set()
    basic        = set()
    def _walk3(t: dict):
        c = t['code'].lower()
        if 'craft' in t:
            if c != target:
                intermediate.add(c)
            for ing in t['craft']['ingredients']:
                _walk3(ing)
        else:
            basic.add(c)
    _walk3(tree)

    buyable = {
        item['code'].lower()
        for item in task['task_info'].get('Resources', [])
        if item.get('subtype') == 'grand_exchange'
    }
    buyable_basic    = basic & buyable
    gatherable_basic = basic - buyable_basic

    for code in intermediate:
        total += _add(breakdown, f'craft_intermediate:{code}')
    for code in gatherable_basic:
        total += _add(breakdown, f'gather_resource:{code}')
    for code in buyable_basic:
        total += _add(breakdown, f'buy_resource:{code}')

    return total, dict(breakdown)


def _worker(code, globals_, locals_, result_queue):
    try:
        exec(code, globals_, locals_)
        result_queue.put(None)
    except Exception as e:
        # send back the exception so we can re-raise in the parent
        result_queue.put(traceback.format_exc())

def safe_exec(code: str, globals_=None, locals_=None, timeout: float = 5.0):
    """
    Executes `code` in a subprocess and kills it if it runs longer than `timeout` seconds.
    Raises a TimeoutError or re-raises any exception from inside the code.
    """
    globals_ = globals_ or {}
    locals_  = locals_  or {}
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(code, globals_, locals_, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        raise TimeoutError(f"Code execution exceeded {timeout} seconds")
    err = q.get()
    if err:
        raise RuntimeError(f"Error in executed code:\n{err}")