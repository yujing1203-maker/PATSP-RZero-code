# -*- coding: utf-8 -*-
"""
Initialize independent LoRA adapters for no-FSDP single-GPU R-Zero reproduction.

This script only creates PEFT LoRA adapters.
It does not train.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--role", required=True, choices=["solver", "challenger"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] base_model = {args.base_model}")
    print(f"[init] out_dir    = {out_dir}")
    print(f"[init] role       = {args.role}")
    print(f"[init] seed       = {args.seed}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))

    meta = {
        "experiment": "rzero_nofsdp_lora_1gpu",
        "mode": "nofsdp_lora_staged",
        "base_model": args.base_model,
        "role": args.role,
        "seed": args.seed,
        "r": args.r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        "note": "Initial LoRA adapter. No GRPO update has been applied yet.",
    }

    meta_path = out_dir / "rzero_nofsdp_lora_meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[save] adapter = {out_dir}")
    print(f"[save] meta    = {meta_path}")


if __name__ == "__main__":
    main()
