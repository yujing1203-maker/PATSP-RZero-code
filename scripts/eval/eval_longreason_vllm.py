import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SYSTEM_PROMPT = (
    "You are a multiple-choice question answering system. "
    "You must answer with exactly one uppercase option letter. "
    "Do not explain. Do not reason step by step. Do not use thinking tags."
)

def unique_sorted_letters(xs):
    seen = []
    for x in xs:
        x = str(x).strip().upper()
        if len(x) == 1 and x.isalpha() and x not in seen:
            seen.append(x)
    return seen

def infer_option_letters(row):
    # Prefer final_inquiry, then question, then prompt.
    candidates = [
        row.get("final_inquiry", ""),
        row.get("question", ""),
        row.get("prompt", ""),
    ]

    letters = []
    for text in candidates:
        if not text:
            continue

        # Match options like:
        # A. xxx
        # B. xxx
        # A) xxx
        found = re.findall(r"(?m)^\s*([A-Z])[\.\)]\s+", str(text))
        letters.extend(found)

    letters = unique_sorted_letters(letters)

    # Keep only plausible MCQ letters.
    letters = [x for x in letters if x in list("ABCDEFG")]

    if letters:
        return letters

    # Fallback: include the gold answer plus A-D.
    gold = str(row.get("answer", "")).strip().upper()
    base = ["A", "B", "C", "D"]
    if gold and gold not in base:
        base.append(gold)
    return base

def format_allowed(letters):
    if len(letters) == 1:
        return letters[0]
    if len(letters) == 2:
        return f"{letters[0]} or {letters[1]}"
    return ", ".join(letters[:-1]) + f", or {letters[-1]}"

def build_user_prompt(row, option_letters):
    prompt = row["prompt"].strip()
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
            return tokenizer.apply_chat_template(
                messages,
                **kwargs,
            )

    return tokenizer.apply_chat_template(
        messages,
        **kwargs,
    )

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

    if t in allowed_set:
        return t

    patterns = [
        rf"FINAL\s+ANSWER\s*[:：]?\s*([{allowed}])\b",
        rf"ANSWER\s*[:：]?\s*([{allowed}])\b",
        rf"OPTION\s*([{allowed}])\b",
        rf"^\s*([{allowed}])[\.\)\s]*$",
        rf"\b([{allowed}])\b",
    ]

    for p in patterns:
        m = re.search(p, t)
        if m:
            return m.group(1)

    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lz1bytedance/LongReason")
    ap.add_argument("--split", default="8k")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.60)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=8)
    ap.add_argument("--skip_too_long", action="store_true")
    ap.add_argument("--enable_thinking", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] dataset:", args.dataset, "split:", args.split, flush=True)
    ds = load_dataset(args.dataset, split=args.split)

    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    print("[load] rows:", len(ds), flush=True)
    print("[load] model:", args.model, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts = []
    rows_meta = []
    skipped_too_long = 0

    option_hist = {}

    for i, row in enumerate(ds):
        option_letters = infer_option_letters(row)
        option_key = "".join(option_letters)
        option_hist[option_key] = option_hist.get(option_key, 0) + 1

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row, option_letters)},
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
            if args.skip_too_long:
                skipped_too_long += 1
                continue
            raise RuntimeError(
                f"Example {i} too long: {input_tokens} tokens > max_model_len={args.max_model_len}. "
                f"Use larger --max_model_len or pass --skip_too_long."
            )

        prompts.append(prompt_text)
        rows_meta.append({
            "row_id": i,
            "example_idx": row.get("example_idx", ""),
            "answer": str(row["answer"]).strip().upper(),
            "input_tokens": input_tokens,
            "option_letters": option_letters,
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

    pred_path = out_dir / "predictions.jsonl"
    correct = 0
    total = 0
    invalid = 0

    with pred_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[start:start + args.batch_size]
            batch_meta = rows_meta[start:start + args.batch_size]

            outputs = llm.generate(batch_prompts, sampling)

            for meta, out in zip(batch_meta, outputs):
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
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "n_total_dataset_rows": len(ds),
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
