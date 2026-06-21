#!/usr/bin/env python3
"""Generate held-out D2 evaluation sets for the four wired benchmarks.

Each env exposes ``sample_new_task(rng, difficulty)``; this script samples a fixed-seed
held-out set at the eval difficulty and serializes ``Task.to_dict()`` (one per line) to
``$RZERO_STORAGE/patsp_eval/<name>.jsonl`` — the canonical names that ``check_env.sh`` /
``scripts/run/eval.sh`` / the self-play drivers expect.

Use this when you do NOT have the original frozen eval sets (they live under
``$RZERO_STORAGE``, not in the code repo). NOTE: this produces an EQUIVALENT held-out set
(same difficulty/structure), NOT the byte-identical frozen set used in the paper — absolute
numbers will differ (different sampled tasks + single-seed noise), but the train-vs-held-out
experiment is valid. For exact paper reproduction, copy the original ``*_D2.jsonl`` over.

Dependencies per env (task generation only):
  - textcraft : pure Python (CPU).
  - alfworld  : pure Python here (its data is needed only at eval time, not to make tasks).
  - herobench : needs the shipped dataset (third_party/HeroBench/datasets/dataset_tasks.json); no server.
  - scienceworld : needs `scienceworld` + java (boots the engine to read task names).
Envs whose deps are missing are skipped with a clear message; the others still get written.

Usage:
  python scripts/data/build_eval_sets.py                       # all four, into $RZERO_STORAGE/patsp_eval
  python scripts/data/build_eval_sets.py --envs textcraft,alfworld
  RZERO_STORAGE=/scratch/$USER/RZero_storage python scripts/data/build_eval_sets.py
"""
import argparse
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_PROJ, "scripts", "rzero_nofsdp_lora"))

# (env, difficulty, n_tasks, output basename) — the canonical sets check_env.sh / _common.sh expect.
SETS = [
    ("textcraft",    0.5, 10, "textcraft_D2.jsonl"),       # depth-3 held-out
    ("textcraft",    0.5, 30, "textcraft_EVAL30.jsonl"),   # larger depth-3 eval set
    ("scienceworld", 0.0, 10, "scienceworld_D2_easy.jsonl"),  # easy end (find-living-thing)
    ("alfworld",     0.5, 10, "alfworld_D2.jsonl"),
    ("herobench",    0.4,  6, "herobench_D2.jsonl"),        # the HEROBENCH_TRAIN_EXCLUDE held-out items
]


def make_env(name):
    if name == "textcraft":
        from envs.textcraft_env import TextCraftEnv
        return TextCraftEnv()
    if name == "scienceworld":
        from envs.scienceworld_env import ScienceWorldEnv
        return ScienceWorldEnv()
    if name == "alfworld":
        from envs.alfworld_env import ALFWorldEnv
        return ALFWorldEnv()
    if name == "herobench":
        from envs.herobench_env import HeroBenchEnv
        return HeroBenchEnv()
    raise ValueError(f"unknown env: {name}")


def _dedupe_key(env_name, task):
    if env_name == "herobench":
        return task.constraints.get("target")
    # Include constraints so tasks that share a spec but differ in the distinguishing field
    # are NOT collapsed (e.g. ALFWorld: same family, distinct per-game `seed`).
    return json.dumps([task.spec, task.constraints], sort_keys=True, ensure_ascii=False)


def _gen_herobench_heldout(env, difficulty, n, rng):
    """HeroBench held-out = the HEROBENCH_TRAIN_EXCLUDE pool indices (disjoint from training).

    Make the env sample ONLY those by temporarily setting the exclude to their complement at
    the eval level. Falls back to plain sampling if the env internals are unavailable.
    """
    held = [int(x) for x in os.environ.get("HEROBENCH_TRAIN_EXCLUDE", "0,4,9,11").replace(" ", "").split(",")
            if x.lstrip("-").isdigit()]
    old = os.environ.get("HEROBENCH_TRAIN_EXCLUDE")
    try:
        lvl = env._level_for(difficulty)          # noqa: SLF001  (tooling read-only access)
        pool_n = len(env._tasks_by_level[lvl])     # noqa: SLF001
        complement = [i for i in range(pool_n) if i not in set(held)]
        if not complement or len(complement) >= pool_n:
            raise RuntimeError("complement empty/degenerate")
        os.environ["HEROBENCH_TRAIN_EXCLUDE"] = ",".join(map(str, complement))
        tasks, seen = [], set()
        for _ in range(pool_n * 6):
            t = env.sample_new_task(random.Random(rng.randint(0, 1 << 30)), difficulty)
            k = _dedupe_key("herobench", t)
            if k not in seen:
                seen.add(k)
                tasks.append(t)
            if len(tasks) >= min(n, len(held)):
                break
        return tasks
    except Exception as ex:  # noqa: BLE001
        print(f"  [warn] herobench held-out-by-index failed ({type(ex).__name__}: {ex}); "
              f"falling back to plain sampling (may overlap training — prefer the original frozen set).")
        return _gen_generic(env, "herobench", difficulty, n, rng)
    finally:
        if old is None:
            os.environ.pop("HEROBENCH_TRAIN_EXCLUDE", None)
        else:
            os.environ["HEROBENCH_TRAIN_EXCLUDE"] = old


def _gen_generic(env, env_name, difficulty, n, rng):
    tasks, seen = [], set()
    for _ in range(n * 10):
        t = env.sample_new_task(random.Random(rng.randint(0, 1 << 30)), difficulty)
        k = _dedupe_key(env_name, t)
        if k not in seen:
            seen.add(k)
            tasks.append(t)
        if len(tasks) >= n:
            break
    return tasks


def gen_tasks(env, env_name, difficulty, n, seed):
    rng = random.Random(seed)
    if env_name == "herobench":
        return _gen_herobench_heldout(env, difficulty, n, rng)
    return _gen_generic(env, env_name, difficulty, n, rng)


def main():
    ap = argparse.ArgumentParser(description="Generate held-out D2 eval sets into $RZERO_STORAGE/patsp_eval/")
    default_out = os.path.join(
        os.environ.get("RZERO_STORAGE", os.path.expanduser("~/RZero_storage")), "patsp_eval")
    ap.add_argument("--out_dir", default=default_out)
    ap.add_argument("--envs", default="textcraft,scienceworld,alfworld,herobench",
                    help="comma-separated subset of {textcraft,scienceworld,alfworld,herobench}")
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    want = {e.strip() for e in args.envs.split(",") if e.strip()}

    # group output specs by env so each env is initialized once
    by_env = {}
    for env_name, diff, n, base in SETS:
        if env_name in want:
            by_env.setdefault(env_name, []).append((diff, n, base))

    print(f"[build_eval_sets] out_dir = {args.out_dir}")
    ok, skipped, failed = 0, 0, 0
    for env_name, specs in by_env.items():
        try:
            env = make_env(env_name)
        except Exception as ex:  # noqa: BLE001
            print(f"[skip] {env_name}: cannot init env ({type(ex).__name__}: {ex}). "
                  f"Install its deps to generate its eval set(s); skipping.")
            skipped += len(specs)
            continue
        for diff, n, base in specs:
            out = os.path.join(args.out_dir, base)
            try:
                tasks = gen_tasks(env, env_name, diff, n, args.seed)
                if not tasks:
                    print(f"[fail] {base}: 0 tasks generated.")
                    failed += 1
                    continue
                with open(out, "w", encoding="utf-8") as f:
                    for t in tasks:
                        f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
                print(f"[ok]   {base}: {len(tasks)} tasks (env={env_name}, difficulty={diff}) -> {out}")
                ok += 1
            except Exception as ex:  # noqa: BLE001
                print(f"[fail] {env_name}/{base}: {type(ex).__name__}: {ex}")
                failed += 1

    print(f"[build_eval_sets] done. ok={ok} skipped={skipped} failed={failed}")
    print("[note] these are EQUIVALENT held-out sets, not the paper's frozen sets; "
          "for exact reproduction copy the original *_D2.jsonl into the out_dir.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
