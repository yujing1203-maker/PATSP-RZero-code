

from __future__ import annotations
import json, math, random
from collections import defaultdict
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_TASKS_FILE   = Path("datasets/dataset_large_tasks.json")
INPUT_PROMPTS_FILE = Path("datasets/dataset_large_prompts.json")
OUTPUT_TASKS_FILE  = Path("datasets/dataset_tasks1.json")
OUTPUT_PROMPTS_FILE= Path("datasets/dataset_prompts2.json")


# ---- Bucket geometry ---------------------------------------------------------
# If you know exactly how many brackets you want, fill NUM_BRACKETS (int)
# and leave BRACKET_DIFFS = None.
# Otherwise leave NUM_BRACKETS = None and set how many *different* difficulties
# each bracket should contain (BRACKET_DIFFS).
NUM_BRACKETS   : int | None = None   # e.g. 10, 12 …  (None = auto)
BRACKET_DIFFS  : int | None = 10     # distinct difficulties per bracket

TASKS_PER_DIFF = 2                   # how many tasks for each difficulty
CRAFT_FRACTION = 1 / 3               # 1 craft : 2 kill  →  0.333…
RANDOM_SEED    = 43                  # set to None for fresh random every run
# ──────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)


# ── helpers ───────────────────────────────────────────────────────────────────
def load_flat() -> list[dict]:
    """Load both source JSONs and flatten them into one list."""
    with INPUT_TASKS_FILE.open(encoding='utf-8') as f:
        tasks_src = json.load(f)
    with INPUT_PROMPTS_FILE.open(encoding='utf-8') as f:
        prompts_src = json.load(f)

    flat: list[dict] = []
    for key, task_list in tasks_src.items():
        for idx, task_obj in enumerate(task_list):
            prompt = prompts_src[key][idx]
            diff   = task_obj.get("total_difficulty", 0)

            # classify type
            is_craft = bool(task_obj.get("task_info", {}).get("Craftable items"))
            ttype    = "craft" if is_craft else "kill"
            monster  = task_obj.get("monster_name")

            flat.append(
                {
                    "data":    task_obj,
                    "prompt":  prompt,
                    "difficulty": diff,
                    "type":    ttype,
                    "monster": monster,
                }
            )
    return flat


def decide_geometry(difficulties: list[int]) -> tuple[int, int, int]:
    """
    Return (num_brackets, diffs_per_bracket, tasks_per_bracket) following config.
    """
    total_diffs = len(difficulties)

    if NUM_BRACKETS is not None and BRACKET_DIFFS is not None:
        raise ValueError("Set either NUM_BRACKETS or BRACKET_DIFFS, not both.")

    if NUM_BRACKETS is None:
        diffs_per_bracket = BRACKET_DIFFS or 1
        num_brackets = math.ceil(total_diffs / diffs_per_bracket)
    else:
        num_brackets = NUM_BRACKETS
        diffs_per_bracket = math.ceil(total_diffs / num_brackets)

    tasks_per_bracket = diffs_per_bracket * TASKS_PER_DIFF
    return num_brackets, diffs_per_bracket, tasks_per_bracket


def evenly_spread(
    flat_tasks: list[dict],
    num_brackets: int,
    diffs_per_bracket: int,
    tasks_per_bracket: int,
) -> list[list[dict]]:
    """
    Core algorithm – fills `num_brackets` buckets with an even difficulty grid.
    """
    # index tasks by difficulty
    diff_map: dict[int, list[dict]] = defaultdict(list)
    for t in flat_tasks:
        diff_map[t["difficulty"]].append(t)

    # sort difficulties so brackets grow in difficulty
    all_diffs = sorted(diff_map.keys())

    brackets: list[list[dict]] = [[] for _ in range(num_brackets)]
    already_used = set()

    for b_idx in range(num_brackets):
        # ----- which difficulties belong to this bracket ---------------------
        start = b_idx * diffs_per_bracket
        end   = start + diffs_per_bracket
        slice_diffs = all_diffs[start:end]

        craft_target = round(tasks_per_bracket * CRAFT_FRACTION)
        kill_target  = tasks_per_bracket - craft_target
        craft_cnt = kill_cnt = 0
        monsters_seen: set[str] = set()

        # ----- first pass: pick TASKS_PER_DIFF fresh tasks for each difficulty
        for diff in slice_diffs:
            candidates = [t for t in diff_map[diff] if id(t) not in already_used]
            random.shuffle(candidates)

            per_diff_needed = TASKS_PER_DIFF
            for t in candidates:
                if per_diff_needed == 0:
                    break

                # monster duplication guard (kill tasks only)
                if (
                    t["type"] == "kill"
                    and t["monster"] is not None
                    and t["monster"] in monsters_seen
                ):
                    # see if another candidate avoids duplication
                    if any(
                        c["monster"] not in monsters_seen
                        for c in candidates
                        if id(c) not in already_used and c["type"] == "kill"
                    ):
                        continue

                # craft / kill soft-quota
                if t["type"] == "craft" and craft_cnt >= craft_target:
                    continue
                if t["type"] == "kill" and kill_cnt >= kill_target:
                    continue

                # accept
                brackets[b_idx].append(t)
                already_used.add(id(t))
                per_diff_needed -= 1
                if t["type"] == "craft":
                    craft_cnt += 1
                else:
                    kill_cnt  += 1
                    if t["monster"]:
                        monsters_seen.add(t["monster"])

        # ----- second pass: if any diff still short, relax constraints -------
        for diff in slice_diffs:
            while (
                sum(1 for t in brackets[b_idx] if t["difficulty"] == diff)
                < TASKS_PER_DIFF
            ):
                # fetch additional tasks (even if duplicates allowed)
                pool = [t for t in diff_map[diff] if id(t) not in already_used]
                if not pool:              # nothing left at this difficulty
                    break
                t = random.choice(pool)   # pick any
                brackets[b_idx].append(t)
                already_used.add(id(t))
                if t["type"] == "craft":
                    craft_cnt += 1
                else:
                    kill_cnt  += 1

        # ----- final pass: strict size top-up (could break ratio) ------------
        while len(brackets[b_idx]) < tasks_per_bracket:
            # try easiest still-empty difficulty first
            for diff in slice_diffs:
                pool = [t for t in diff_map[diff] if id(t) not in already_used]
                if pool:
                    t = pool.pop()
                    brackets[b_idx].append(t)
                    already_used.add(id(t))
                    break
            else:
                # nothing left at *any* of this bracket's difficulties
                break

        # sort inside bracket
        brackets[b_idx].sort(key=lambda x: x["difficulty"])

    return brackets


def export(brackets: list[list[dict]]):
    """Write the two new JSON files mirroring the source structure."""
    tasks_out   = {str(i + 1): [t["data"]   for t in b] for i, b in enumerate(brackets)}
    prompts_out = {str(i + 1): [t["prompt"] for t in b] for i, b in enumerate(brackets)}

    with OUTPUT_TASKS_FILE.open("w", encoding="utf-8") as f:
        json.dump(tasks_out, f, indent=2, ensure_ascii=False)
    with OUTPUT_PROMPTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(prompts_out, f, indent=2, ensure_ascii=False)
    print(f"✅  Wrote {OUTPUT_TASKS_FILE} and {OUTPUT_PROMPTS_FILE}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    flat = load_flat()
    all_diffs = sorted({t["difficulty"] for t in flat})

    num_brackets, diffs_per_bracket, tasks_per_bracket = decide_geometry(all_diffs)

    print(
        f"Building {num_brackets} brackets • "
        f"{diffs_per_bracket} diffs/bracket • "
        f"{tasks_per_bracket} tasks/bracket "
        f"(craft≈{round(tasks_per_bracket*CRAFT_FRACTION)})"
    )

    brackets = evenly_spread(flat, num_brackets, diffs_per_bracket, tasks_per_bracket)
    export(brackets)


if __name__ == "__main__":
    main()