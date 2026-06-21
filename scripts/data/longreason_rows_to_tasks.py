import argparse
import json
from pathlib import Path


def convert_row(row):
    exid = str(row["example_idx"])
    source_split = str(row.get("source_split", "8k"))
    option_letters = [str(x).strip().upper() for x in row.get("option_letters", [])]
    answer = str(row["answer"]).strip().upper()

    if answer and answer not in option_letters:
        option_letters.append(answer)

    task_id = str(row.get("id") or f"longreason_{source_split}_{exid}")

    return {
        "task_id": task_id,
        "spec": {
            "goal": "answer the LongReason multiple-choice question",
            "prompt": str(row["prompt"]),
        },
        "constraints": {
            "answer": answer,
            "option_letters": option_letters,
            "example_idx": exid,
            "source_split": source_split,
        },
        "info": {
            "env": "longreason",
            "dataset": row.get("dataset", "lz1bytedance/LongReason"),
            "input_tokens": int(row.get("input_tokens", 0) or 0),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with in_path.open("r", encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task = convert_row(row)
            g.write(json.dumps(task, ensure_ascii=False) + "\n")
            n += 1

    print(f"wrote {n} tasks -> {out_path}")


if __name__ == "__main__":
    main()
