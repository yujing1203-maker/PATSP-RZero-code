import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def unique_sorted_letters(xs):
    seen = []
    for x in xs:
        x = str(x).strip().upper()
        if len(x) == 1 and x.isalpha() and x not in seen:
            seen.append(x)
    return seen


def infer_option_letters(row):
    candidates = [
        row.get("final_inquiry", ""),
        row.get("question", ""),
        row.get("prompt", ""),
    ]

    letters = []
    for text in candidates:
        if not text:
            continue
        found = re.findall(r"(?m)^\s*([A-Z])[\.\)]\s+", str(text))
        letters.extend(found)

    letters = unique_sorted_letters(letters)
    letters = [x for x in letters if x in list("ABCDEFG")]

    gold = str(row.get("answer", "")).strip().upper()

    if letters:
        if gold and gold not in letters:
            # Do not silently drop a valid gold label.
            letters.append(gold)
        return unique_sorted_letters(letters)

    # Fallback.
    base = ["A", "B", "C", "D"]
    if gold and gold not in base:
        base.append(gold)
    return base


def build_prompt(row):
    return str(row["prompt"]).strip()


def count_tokens(tokenizer, prompt):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a multiple-choice question answering system. "
                "You must answer with exactly one uppercase option letter."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    try:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

    return len(ids)


def normalize_row(row, source_split, tokenizer):
    example_idx = str(row["example_idx"])
    answer = str(row["answer"]).strip().upper()
    option_letters = infer_option_letters(row)
    prompt = build_prompt(row)

    return {
        "env": "longreason",
        "dataset": "lz1bytedance/LongReason",
        "source_split": source_split,
        "example_idx": example_idx,
        "id": f"longreason_{source_split}_{example_idx}",
        "prompt": prompt,
        "answer": answer,
        "option_letters": option_letters,
        "question": row.get("question", ""),
        "background": row.get("background", ""),
        "final_inquiry": row.get("final_inquiry", ""),
        "analysis": row.get("analysis", ""),
        "input_tokens": count_tokens(tokenizer, prompt),
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lz1bytedance/LongReason")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=20260618)
    ap.add_argument("--eval_frac", type=float, default=0.30)
    ap.add_argument("--primary_split", default="8k")
    ap.add_argument("--aux_splits", nargs="+", default=["original", "expanded"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    all_splits = [args.primary_split] + list(args.aux_splits)

    print("[load] dataset:", args.dataset)
    print("[load] splits:", all_splits)

    split_rows = {}
    split_by_id = {}

    for sp in all_splits:
        ds = load_dataset(args.dataset, split=sp)
        rows = [normalize_row(row, sp, tokenizer) for row in ds]
        split_rows[sp] = rows
        split_by_id[sp] = {r["example_idx"]: r for r in rows}

        print(f"[split] {sp}: rows={len(rows)} unique_ids={len(split_by_id[sp])}")

    primary_ids = sorted(split_by_id[args.primary_split].keys())

    # Sanity: every aux split should contain the primary IDs.
    for sp in all_splits:
        ids = set(split_by_id[sp].keys())
        missing = sorted(set(primary_ids) - ids)
        extra = sorted(ids - set(primary_ids))
        print(f"[check] {sp}: missing_vs_primary={len(missing)} extra_vs_primary={len(extra)}")
        if missing:
            print("[warn] first missing:", missing[:5])
        if extra:
            print("[warn] first extra:", extra[:5])

    rng = random.Random(args.seed)
    ids = list(primary_ids)
    rng.shuffle(ids)

    n_eval = int(round(len(ids) * args.eval_frac))
    eval_ids = set(ids[:n_eval])
    train_ids = set(ids[n_eval:])

    print("[partition] total_ids:", len(ids))
    print("[partition] train_ids:", len(train_ids))
    print("[partition] eval_ids:", len(eval_ids))
    print("[partition] overlap:", len(train_ids & eval_ids))

    assert len(train_ids & eval_ids) == 0

    meta = {
        "dataset": args.dataset,
        "seed": args.seed,
        "eval_frac": args.eval_frac,
        "primary_split": args.primary_split,
        "aux_splits": args.aux_splits,
        "n_total_ids": len(ids),
        "n_train_ids": len(train_ids),
        "n_eval_ids": len(eval_ids),
    }

    with (out_dir / "longreason_split_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Primary train/eval.
    primary = args.primary_split
    primary_train = [split_by_id[primary][i] for i in sorted(train_ids)]
    primary_eval = [split_by_id[primary][i] for i in sorted(eval_ids)]

    write_jsonl(out_dir / f"longreason_{primary}_train.jsonl", primary_train)
    write_jsonl(out_dir / f"longreason_{primary}_eval.jsonl", primary_eval)

    # Aux eval only, same held-out IDs.
    for sp in args.aux_splits:
        aux_eval = [split_by_id[sp][i] for i in sorted(eval_ids) if i in split_by_id[sp]]
        write_jsonl(out_dir / f"longreason_{sp}_eval.jsonl", aux_eval)

    # Validation report.
    for path in sorted(out_dir.glob("*.jsonl")):
        n = 0
        bad_answer = 0
        max_tokens = 0
        option_hist = {}

        with path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                n += 1
                max_tokens = max(max_tokens, int(r["input_tokens"]))
                key = "".join(r["option_letters"])
                option_hist[key] = option_hist.get(key, 0) + 1
                if r["answer"] not in r["option_letters"]:
                    bad_answer += 1

        print(
            "[file]",
            path.name,
            "rows=", n,
            "bad_answer=", bad_answer,
            "max_input_tokens=", max_tokens,
            "option_hist=", json.dumps(option_hist, sort_keys=True),
        )

    print("[done] out_dir:", out_dir)


if __name__ == "__main__":
    main()
