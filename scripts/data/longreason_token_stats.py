import argparse
import json
import statistics
from datasets import load_dataset
from transformers import AutoTokenizer

SYSTEM_PROMPT = "You are a careful long-context multiple-choice reasoning assistant."

def build_user_prompt(row):
    prompt = row["prompt"].strip()
    return (
        prompt
        + "\n\n"
        + "Return the final answer as a single uppercase letter from A, B, C, or D. "
          "Do not include explanation."
    )

def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = int(round((len(xs) - 1) * p / 100))
    return xs[k]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lz1bytedance/LongReason")
    ap.add_argument("--splits", nargs="+", default=["original", "expanded", "8k"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_model_len", type=int, default=8192)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    for sp in args.splits:
        print("\n" + "=" * 100)
        print("split:", sp)

        ds = load_dataset(args.dataset, split=sp)
        if args.limit is not None:
            ds = ds.select(range(min(args.limit, len(ds))))

        lengths = []
        too_long = 0

        for row in ds:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(row)},
            ]

            ids = tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )

            n = len(ids)
            lengths.append(n)
            if n > args.max_model_len:
                too_long += 1

        print("n:", len(lengths))
        print("min:", min(lengths))
        print("p50:", percentile(lengths, 50))
        print("p90:", percentile(lengths, 90))
        print("p95:", percentile(lengths, 95))
        print("p99:", percentile(lengths, 99))
        print("max:", max(lengths))
        print(f"over_{args.max_model_len}:", too_long)

        print("summary_json:", json.dumps({
            "split": sp,
            "n": len(lengths),
            "min": min(lengths),
            "p50": percentile(lengths, 50),
            "p90": percentile(lengths, 90),
            "p95": percentile(lengths, 95),
            "p99": percentile(lengths, 99),
            "max": max(lengths),
            f"over_{args.max_model_len}": too_long,
        }, ensure_ascii=False))

if __name__ == "__main__":
    main()
