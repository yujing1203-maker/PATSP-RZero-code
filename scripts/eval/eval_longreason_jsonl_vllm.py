import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = (
    "You are a multiple-choice question answering system. "
    "You must answer with exactly one uppercase option letter. "
    "Do not explain. Do not reason step by step. Do not use thinking tags."
)


def format_allowed(letters):
    if len(letters) == 1:
        return letters[0]
    if len(letters) == 2:
        return f"{letters[0]} or {letters[1]}"
    return ", ".join(letters[:-1]) + f", or {letters[-1]}"


def build_user_prompt(row):
    prompt = row["prompt"].strip()
    option_letters = row["option_letters"]
    allowed = format_allowed(option_letters)

    return (
        prompt
        + "\n\n"
        + f"IMPORTANT: Output exactly one character from {allowed}. "
          "No words, no punctuation, no explanation."
    )


def apply_chat_template(tokenizer, messages, tokenize, enable_thinking):
    kwargs = {
        "tokenize": tokenize,
        "add_generation_prompt": True,
    }

    if not enable_thinking:
        try:
            return tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)

    return tokenizer.apply_chat_template(messages, **kwargs)


def extract_answer(text, option_letters):
    if text is None:
        return None

    raw = text.strip()
    if not raw:
        return None

    allowed = "".join(option_letters)
    allowed_set = set(option_letters)

    t = raw.upper().strip()
    t = t.replace("<THINK>", " ").replace("</THINK>", " ")

    # Exact one-letter output.
    if t in allowed_set:
        return t

    # Strong explicit patterns.
    patterns = [
        rf"FINAL\s+ANSWER\s*[:：]?\s*([{allowed}])\b",
        rf"ANSWER\s*[:：]?\s*([{allowed}])\b",
        rf"OPTION\s*([{allowed}])\b",
        rf"^\s*([{allowed}])[\.\)\s]*$",
    ]

    for p in patterns:
        m = re.search(p, t)
        if m:
            return m.group(1)

    # Do NOT use a broad \b([A-G])\b fallback here.
    # It can incorrectly capture variables in reasoning text.

    return None


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.60)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=8)
    ap.add_argument("--enable_thinking", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.jsonl)
    if args.limit is not None:
        rows = rows[:args.limit]

    print("[load] jsonl:", args.jsonl, flush=True)
    print("[load] rows:", len(rows), flush=True)
    print("[load] model:", args.model, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts = []
    metas = []
    skipped_too_long = 0
    option_hist = {}

    for i, row in enumerate(rows):
        option_letters = [str(x).strip().upper() for x in row["option_letters"]]
        key = "".join(option_letters)
        option_hist[key] = option_hist.get(key, 0) + 1

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row)},
        ]

        prompt_text = apply_chat_template(
            tokenizer,
            messages,
            tokenize=False,
            enable_thinking=args.enable_thinking,
        )

        token_ids = apply_chat_template(
            tokenizer,
            messages,
            tokenize=True,
            enable_thinking=args.enable_thinking,
        )

        input_tokens = len(token_ids)

        if input_tokens > args.max_model_len:
            skipped_too_long += 1
            continue

        prompts.append(prompt_text)
        metas.append({
            "row_id": i,
            "id": row.get("id", ""),
            "example_idx": row["example_idx"],
            "source_split": row.get("source_split", ""),
            "answer": str(row["answer"]).strip().upper(),
            "option_letters": option_letters,
            "input_tokens": input_tokens,
        })

    print("[prepare] prompts:", len(prompts), flush=True)
    print("[prepare] skipped_too_long:", skipped_too_long, flush=True)
    print("[prepare] option_hist:", json.dumps(option_hist, ensure_ascii=False, sort_keys=True), flush=True)

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
    )

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )

    correct = 0
    total = 0
    invalid = 0

    pred_path = out_dir / "predictions.jsonl"

    with pred_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[start:start + args.batch_size]
            batch_metas = metas[start:start + args.batch_size]

            outputs = llm.generate(batch_prompts, sampling)

            for meta, out in zip(batch_metas, outputs):
                text = out.outputs[0].text if out.outputs else ""
                pred = extract_answer(text, meta["option_letters"])
                gold = meta["answer"]

                is_correct = pred == gold
                if pred is None:
                    invalid += 1
                if is_correct:
                    correct += 1
                total += 1

                rec = {
                    **meta,
                    "prediction": pred,
                    "gold": gold,
                    "correct": bool(is_correct),
                    "raw_output": text,
                }

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            acc = correct / total if total else 0.0
            print(f"[progress] {total}/{len(prompts)} acc={acc:.4f} invalid={invalid}", flush=True)

    summary = {
        "jsonl": args.jsonl,
        "model": args.model,
        "n_total_rows": len(rows),
        "n_evaluated": total,
        "skipped_too_long": skipped_too_long,
        "correct": correct,
        "invalid": invalid,
        "accuracy": correct / total if total else None,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
        "option_hist": option_hist,
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[summary]", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("[out_dir]", out_dir, flush=True)


if __name__ == "__main__":
    main()
