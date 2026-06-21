#!/usr/bin/env python3
"""
Build a multi-checkpoint TRAINING parquet.

One row per problem with the FROZEN schema:
    problem:        str   # the question text
    answer:         str   # final gold answer
    checkpoints:    str   # JSON string of list[{id,gold,order,source}]
    n_checkpoints:  int
    source:         str
    verification:   str
    note:           str

GSM8K checkpoint extraction (the cheap correct trick): GSM8K gold solutions embed
calculator annotations <<a op b = c>>. We parse them IN ORDER -> each c is an ordered
intermediate gold checkpoint. The final #### value is the last checkpoint.

MATH-500: single checkpoint = final boxed answer (n_checkpoints=1).

Load / normalize / parquet-write conventions are reused from the sibling builder
scripts/rzero_nofsdp_lora/build_external_math_eval_parquets.py (same HF ids, same
fallbacks, same take_subset / save_parquet helpers).
"""

import argparse
import json
import random
import re
import traceback
from pathlib import Path

from datasets import Dataset, load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["gsm8k", "math500"])
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out_parquet", required=True)
    p.add_argument("--seed", type=int, default=3000)
    return p.parse_args()


# ------------------------------------------------------------------ loaders ---
# Mirror build_external_math_eval_parquets.py exactly (ids + fallbacks + order).

def load_gsm8k():
    errors = []
    for name, subset in [
        ("openai/gsm8k", "main"),
        ("gsm8k", "main"),
    ]:
        try:
            return load_dataset(name, subset, split="train", trust_remote_code=True)
        except Exception as e:
            errors.append(f"{name}/{subset}: {repr(e)}")
    raise RuntimeError("Failed to load GSM8K:\n" + "\n".join(errors))


def load_math500():
    errors = []
    attempts = [
        ("HuggingFaceH4/MATH-500", None, "test"),
        ("lighteval/MATH", "all", "test"),
    ]
    for name, subset, split in attempts:
        try:
            if subset is None:
                return load_dataset(name, split=split, trust_remote_code=True)
            return load_dataset(name, subset, split=split, trust_remote_code=True)
        except Exception as e:
            errors.append(f"{name}/{subset}/{split}: {repr(e)}")
    raise RuntimeError("Failed to load MATH-500/MATH fallback:\n" + "\n".join(errors))


def take_subset(ds, n, seed):
    n = min(n, len(ds))
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    return [ds[i] for i in idxs[:n]]


# --------------------------------------------------------------- normalize ---

def gsm8k_final_answer(answer_text: str) -> str:
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", answer_text)
    return nums[-1] if nums else answer_text.strip()


def normalize_math_answer(row):
    for key in ["answer", "final_answer", "target"]:
        if key in row and row[key] is not None:
            return str(row[key]).strip()

    if "solution" in row and row["solution"] is not None:
        sol = str(row["solution"])
        boxed = re.findall(r"\\boxed\{([^{}]+)\}", sol)
        if boxed:
            return boxed[-1].strip()

    return ""


# GSM8K calculator annotations: <<a op b = c>>. We only need the post-"=" value c.
_CALC_ANNOTATION = re.compile(r"<<\s*[^<>]*?=\s*([^<>=]+?)\s*>>")


def parse_gsm8k_checkpoints(answer_text: str, final_answer: str):
    """
    Parse calculator annotations in order. Each annotation's right-hand value c is
    an ordered intermediate gold checkpoint. The final #### value is appended as
    the last checkpoint (deduplicated if the last annotation already equals it).
    """
    raw_values = [m.strip() for m in _CALC_ANNOTATION.findall(answer_text)]
    raw_values = [v for v in raw_values if v != ""]

    golds = list(raw_values)

    final = (final_answer or "").strip()
    if final:
        if not golds or golds[-1] != final:
            golds.append(final)

    checkpoints = []
    for order, gold in enumerate(golds):
        checkpoints.append(
            {
                "id": f"cp{order}",
                "gold": gold,
                "order": order,
                "source": "gsm8k_calc_annotation",
            }
        )
    return checkpoints


# ------------------------------------------------------------------- build ---

def build_gsm8k_rows(n, seed):
    ds = load_gsm8k()
    rows = []
    for i, r in enumerate(take_subset(ds, n, seed)):
        q = str(r.get("question", "")).strip()
        raw_answer = str(r.get("answer", "")).strip()
        final = gsm8k_final_answer(raw_answer)
        if not q or not final:
            continue

        checkpoints = parse_gsm8k_checkpoints(raw_answer, final)
        if not checkpoints:
            # No annotations parsed; fall back to a single final checkpoint.
            checkpoints = [
                {
                    "id": "cp0",
                    "gold": final,
                    "order": 0,
                    "source": "gsm8k_calc_annotation",
                }
            ]

        rows.append(
            {
                "problem": q,
                "answer": final,
                "checkpoints": json.dumps(checkpoints, ensure_ascii=False),
                "n_checkpoints": len(checkpoints),
                "source": "multistep_gsm8k_train_subset",
                "verification": "gsm8k_calc_annotation_checkpoints",
                "note": f"dataset=gsm8k; subset_index={i}; seed={seed}",
            }
        )
    return rows


def build_math500_rows(n, seed):
    ds = load_math500()
    rows = []
    for i, r in enumerate(take_subset(ds, n, seed)):
        q = str(r.get("problem") or r.get("question") or "").strip()
        a = normalize_math_answer(r)
        if not q or not a:
            continue

        checkpoints = [
            {
                "id": "cp0",
                "gold": a,
                "order": 0,
                "source": "math500_final_boxed",
            }
        ]

        rows.append(
            {
                "problem": q,
                "answer": a,
                "checkpoints": json.dumps(checkpoints, ensure_ascii=False),
                "n_checkpoints": 1,
                "source": "multistep_math500_train_subset",
                "verification": "math500_single_checkpoint",
                "note": f"dataset=math500; subset_index={i}; seed={seed}",
            }
        )
    return rows


def save_parquet(rows, path: Path):
    ds = Dataset.from_list(rows)
    ds.to_parquet(str(path))
    return len(ds)


def main():
    args = parse_args()

    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset == "gsm8k":
        rows = build_gsm8k_rows(args.n, args.seed)
    else:
        rows = build_math500_rows(args.n, args.seed)

    if not rows:
        raise RuntimeError(f"{args.dataset} produced zero usable rows.")

    count = save_parquet(rows, out_path)

    n_cps = [r["n_checkpoints"] for r in rows]
    summary = {
        "dataset": args.dataset,
        "requested_n": args.n,
        "seed": args.seed,
        "out_parquet": str(out_path),
        "rows": count,
        "n_checkpoints_min": min(n_cps),
        "n_checkpoints_max": max(n_cps),
        "n_checkpoints_mean": sum(n_cps) / len(n_cps),
        "multi_checkpoint_rows": sum(1 for c in n_cps if c > 1),
        "schema": [
            "problem",
            "answer",
            "checkpoints",
            "n_checkpoints",
            "source",
            "verification",
            "note",
        ],
    }

    summary_path = out_path.with_name(out_path.stem + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[save] parquet = {out_path}")
    print(f"[save] summary = {summary_path}")


if __name__ == "__main__":
    main()
