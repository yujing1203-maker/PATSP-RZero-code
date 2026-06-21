# -*- coding: utf-8 -*-
"""
Train a PEFT LoRA adapter from GRPO-scored rollouts.

No FSDP.
No Ray.
No vLLM during training.

This is the single-GPU staged trainer for R-Zero LoRA reproduction.

Objective:
  PPO/GRPO clipped policy objective on response tokens.

Implemented as:
  1. precompute old_log_probs using the initial LoRA policy
  2. train with ratio = exp(new_log_probs - old_log_probs)
  3. apply clipped policy loss compatible with verl.trainer.core_algos.compute_policy_loss

For valid challenger samples, we optionally canonicalize the training completion:
  <question>
  ...
  </question>
  \\boxed{...}

This prevents reinforcing dirty prefixes such as <translation>, <location>, etc.
"""

import argparse
import json
import math
import os
import random
import functools
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_in", default=None,
                        help="LoRA adapter to continue training (LoRA mode). Omit when --full_param.")
    parser.add_argument("--full_param", action="store_true",
                        help="Full-parameter fine-tune the base model (no PEFT wrap; all weights "
                             "trainable). Same GRPO objective/rollouts as LoRA mode — for the "
                             "LoRA-vs-full bridge study. 1.5B fits on one 97GB card.")
    parser.add_argument("--rollouts_jsonl", required=True)
    parser.add_argument("--lora_out", required=True, help="Output directory. Legacy name; in --full_param/FSDP mode this is a full checkpoint dir, not a LoRA adapter dir.")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_rows", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--min_abs_adv", type=float, default=1e-8)

    parser.add_argument(
        "--objective",
        choices=["pg", "grpo_clip"],
        default="grpo_clip",
        help="pg keeps the old simple objective; grpo_clip uses PPO/GRPO clipped objective.",
    )
    parser.add_argument("--clip_ratio_low", type=float, default=0.2)
    parser.add_argument("--clip_ratio_high", type=float, default=0.3)
    parser.add_argument("--clip_ratio_dual", type=float, default=3.0)
    parser.add_argument(
        "--old_logprob_mode",
        choices=["precompute"],
        default="precompute",
        help="Precompute old log-probs with the initial policy before optimizer updates.",
    )

    parser.add_argument(
        "--no_canonicalize_valid",
        action="store_true",
        help="Use raw completion_for_train even for valid rows. Not recommended for early Challenger training.",
    )
    parser.add_argument(
        "--credit_mode",
        choices=["uniform", "outcome_only", "stage_hardcoded", "routed"],
        default="uniform",
        help=(
            "uniform = original scalar-broadcast GRPO (default, behavior-preserving). "
            "outcome_only/stage_hardcoded/routed consume per-stage token_advantages built "
            "from verifier stage_records + stage_advantages on each rollout row."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def canonical_completion(row: dict) -> str:
    question = (row.get("question") or "").strip()
    answer = (row.get("answer") or "").strip()
    return f"<question>\n{question}\n</question>\n\\boxed{{{answer}}}"


def choose_completion(row: dict, canonicalize_valid: bool) -> str:
    if canonicalize_valid and row.get("format_ok") and row.get("question") and row.get("answer"):
        return canonical_completion(row)

    return (row.get("completion_for_train") or row.get("completion") or "").strip()


def trainable_abs_sum(model) -> float:
    total = 0.0
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                total += p.detach().float().abs().sum().item()
    return total


def count_trainable_params(model) -> Tuple[int, int]:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def build_token_advantages(row: dict, completion_ids, offsets, credit_mode: str):
    """Per-completion-token advantage vector, or None to fall back to scalar broadcast.

    `offsets` is the list of (char_start, char_end) aligned to `completion_ids`
    (from tokenizer return_offsets_mapping). Verifier stage spans (char_span on the
    completion text) are mapped onto the tokens they cover; tokens outside every
    stage span receive the per-sequence outcome advantage A_T.
    """
    if credit_mode == "uniform":
        return None
    stage_records = row.get("stage_records") or []
    stage_advs = row.get("stage_advantages")
    if not stage_records or stage_advs is None or len(stage_advs) != len(stage_records):
        return None
    outcome_adv = float(row.get("outcome_advantage", row.get("advantage", 0.0)))
    tvec = [outcome_adv] * len(completion_ids)
    for rec, sadv in zip(stage_records, stage_advs):
        span = rec.get("char_span") if isinstance(rec, dict) else None
        if not span or len(span) != 2:
            continue
        s, e = int(span[0]), int(span[1])
        for ti, off in enumerate(offsets):
            if off is None:
                continue
            cs, ce = int(off[0]), int(off[1])
            if ce > cs and cs >= s and ce <= e:
                tvec[ti] = float(sadv)
    return tvec


def build_example(tokenizer, row: dict, max_seq_len: int, canonicalize_valid: bool, credit_mode: str = "uniform"):
    prompt = row.get("prompt") or ""
    completion = choose_completion(row, canonicalize_valid=canonicalize_valid)

    if not prompt.strip() or not completion.strip():
        return None

    if tokenizer.eos_token is not None and not completion.endswith(tokenizer.eos_token):
        completion = completion + tokenizer.eos_token

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    ).input_ids

    # In stage-credit modes we need char->token offsets to map verifier stage spans
    # onto completion tokens. Fall back to scalar (uniform) if offsets are unavailable.
    token_advs = None
    if credit_mode != "uniform":
        try:
            enc = tokenizer(
                completion,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            completion_ids = enc["input_ids"]
            offsets = enc["offset_mapping"]
            token_advs = build_token_advantages(row, completion_ids, offsets, credit_mode)
        except Exception as ex:  # noqa: BLE001
            print(
                f"[warn] offset/stage-advantage build failed uid={row.get('uid')} "
                f"sample={row.get('sample_index')}: {ex}; falling back to scalar advantage"
            )
            completion_ids = tokenizer(completion, add_special_tokens=False).input_ids
            token_advs = None
    else:
        completion_ids = tokenizer(
            completion,
            add_special_tokens=False,
        ).input_ids

    if not prompt_ids or not completion_ids:
        return None

    total_len = len(prompt_ids) + len(completion_ids)
    if total_len > max_seq_len:
        return None

    if token_advs is not None and len(token_advs) != len(completion_ids):
        # alignment mismatch -> safe fallback to scalar broadcast
        token_advs = None

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    attention_mask = [1] * len(input_ids)

    return {
        "uid": row.get("uid"),
        "sample_index": row.get("sample_index"),
        "advantage": float(row.get("advantage", 0.0)),
        "reward": float(row.get("reward", 0.0)),
        "format_ok": bool(row.get("format_ok")),
        "solver_score": float(row.get("solver_score", 0.0)),
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "prompt_len": len(prompt_ids),
        "completion_len": len(completion_ids),
        "completion_text": completion,
        "token_advantages": (
            torch.tensor(token_advs, dtype=torch.float) if token_advs is not None else None
        ),
    }


def extract_response_token_log_probs(model, example: Dict, device: torch.device, use_grad: bool):
    input_ids = example["input_ids"].unsqueeze(0).to(device)
    attention_mask = example["attention_mask"].unsqueeze(0).to(device)
    labels = example["labels"].unsqueeze(0).to(device)

    context = torch.enable_grad() if use_grad else torch.no_grad()
    with context:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        logits = outputs.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        mask = shift_labels.ne(-100)
        ntokens = int(mask.sum().item())
        if ntokens == 0:
            return None, 0

        # Only compute log-softmax on response-token positions to reduce memory.
        masked_logits = shift_logits[mask]
        masked_labels = shift_labels[mask]

        log_probs = F.log_softmax(masked_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1,
            index=masked_labels.unsqueeze(-1),
        ).squeeze(-1)

        return token_log_probs, ntokens


def compute_pg_loss(model, example: Dict, device: torch.device):
    token_log_probs, ntokens = extract_response_token_log_probs(
        model=model,
        example=example,
        device=device,
        use_grad=True,
    )
    if token_log_probs is None:
        return None, 0, {}

    token_advs = example.get("token_advantages")
    if token_advs is not None and token_advs.numel() == token_log_probs.numel():
        adv = token_advs.to(device=device, dtype=token_log_probs.dtype)
        loss = -(adv * token_log_probs).sum() / token_log_probs.numel()
        info = {
            "mean_logprob": float(token_log_probs.mean().detach().float().cpu()),
            "per_token_adv": True,
        }
        return loss, ntokens, info

    mean_logprob = token_log_probs.mean()
    advantage = torch.tensor(
        float(example["advantage"]),
        device=device,
        dtype=mean_logprob.dtype,
    )

    loss = -advantage * mean_logprob
    info = {
        "mean_logprob": float(mean_logprob.detach().float().cpu()),
        "per_token_adv": False,
    }
    return loss, ntokens, info


@torch.no_grad()
def precompute_old_log_probs(model, examples: List[Dict], device: torch.device):
    was_training = model.training
    model.eval()

    ok = 0
    skipped = 0
    total_tokens = 0

    print("[old] precomputing old_log_probs with initial policy")

    for i, ex in enumerate(examples):
        try:
            old_token_log_probs, ntokens = extract_response_token_log_probs(
                model=model,
                example=ex,
                device=device,
                use_grad=False,
            )
        except torch.cuda.OutOfMemoryError:
            old_token_log_probs = None
            ntokens = 0
            torch.cuda.empty_cache()

        if old_token_log_probs is None:
            ex["old_token_log_probs"] = None
            skipped += 1
            print(f"[old][skip] row={i} uid={ex['uid']} sample={ex['sample_index']}")
            continue

        ex["old_token_log_probs"] = old_token_log_probs.detach().float().cpu()
        ok += 1
        total_tokens += ntokens

    if was_training:
        model.train()

    print(f"[old] precompute ok={ok} skipped={skipped} total_response_tokens={total_tokens}")
    return ok, skipped, total_tokens


def compute_grpo_clip_loss(
    model,
    example: Dict,
    device: torch.device,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_dual: float,
):
    old_token_log_probs = example.get("old_token_log_probs")
    if old_token_log_probs is None:
        return None, 0, {}

    new_token_log_probs, ntokens = extract_response_token_log_probs(
        model=model,
        example=example,
        device=device,
        use_grad=True,
    )
    if new_token_log_probs is None:
        return None, 0, {}

    old_token_log_probs = old_token_log_probs.to(
        device=device,
        dtype=new_token_log_probs.dtype,
    )

    if old_token_log_probs.numel() != new_token_log_probs.numel():
        return None, 0, {}

    # This is token-level equivalent of verl.trainer.core_algos.compute_policy_loss.
    negative_approx_kl = new_token_log_probs - old_token_log_probs

    # Raw ratio is clamped only for numerical safety.
    ratio = torch.exp(torch.clamp(negative_approx_kl, min=-20.0, max=20.0))

    low = max(min(float(clip_ratio_low), 0.99), 0.0)
    high = max(float(clip_ratio_high), 0.0)

    clipped_ratio = torch.exp(
        torch.clamp(
            negative_approx_kl,
            math.log(1.0 - low),
            math.log(1.0 + high),
        )
    )

    token_advs = example.get("token_advantages")
    if token_advs is not None and token_advs.numel() == new_token_log_probs.numel():
        advantages = token_advs.to(device=device, dtype=new_token_log_probs.dtype)
    else:
        advantages = torch.full_like(
            new_token_log_probs,
            fill_value=float(example["advantage"]),
        )

    pg_loss = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    pg_loss3 = -advantages * float(clip_ratio_dual)

    clipped_pg_loss_higher = torch.maximum(pg_loss, pg_loss2)
    pg_clipfrac_higher = (pg_loss < pg_loss2).float()

    clipped_pg_loss_lower = torch.minimum(clipped_pg_loss_higher, pg_loss3)
    final_pg_loss = torch.where(
        advantages < 0,
        clipped_pg_loss_lower,
        clipped_pg_loss_higher,
    )
    pg_clipfrac_lower = (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float()

    loss = final_pg_loss.mean()

    info = {
        "ppo_kl": float((-negative_approx_kl).mean().detach().float().cpu()),
        "pg_clipfrac_higher": float(pg_clipfrac_higher.mean().detach().float().cpu()),
        "pg_clipfrac_lower": float(pg_clipfrac_lower.mean().detach().float().cpu()),
        "ratio_mean": float(ratio.mean().detach().float().cpu()),
        "ratio_min": float(ratio.min().detach().float().cpu()),
        "ratio_max": float(ratio.max().detach().float().cpu()),
        "logprob_delta_mean": float(negative_approx_kl.mean().detach().float().cpu()),
    }
    return loss, ntokens, info


def compute_training_loss(model, example: Dict, device: torch.device, args):
    if args.objective == "pg":
        return compute_pg_loss(model, example, device=device)

    if args.objective == "grpo_clip":
        return compute_grpo_clip_loss(
            model=model,
            example=example,
            device=device,
            clip_ratio_low=args.clip_ratio_low,
            clip_ratio_high=args.clip_ratio_high,
            clip_ratio_dual=args.clip_ratio_dual,
        )

    raise ValueError(f"Unknown objective: {args.objective}")



def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _dist_ready() else 0


def _world_size() -> int:
    return dist.get_world_size() if _dist_ready() else 1


def _is_rank0() -> bool:
    return _rank() == 0

def main():
    args = parse_args()

    # FSDP is enabled only when this script is launched by torchrun.
    # Example:
    #   CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.run --standalone --nproc_per_node 2 ...
    fsdp_enabled = args.full_param and ("LOCAL_RANK" in os.environ)

    if fsdp_enabled:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if rank == 0:
            print(f"[fsdp] enabled world_size={world_size}", flush=True)
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    canonicalize_valid = not args.no_canonicalize_valid

    rollouts_path = Path(args.rollouts_jsonl)
    lora_out = Path(args.lora_out)
    lora_out.mkdir(parents=True, exist_ok=True)

    raw_rows = load_jsonl(rollouts_path)

    selected_rows = []
    for row in raw_rows:
        adv = float(row.get("advantage", 0.0))
        keep = abs(adv) > args.min_abs_adv
        if not keep and args.credit_mode in ("routed", "stage_hardcoded", "outcome_only"):
            sadvs = row.get("stage_advantages") or []
            if any(abs(float(x)) > args.min_abs_adv for x in sadvs):
                keep = True
        if keep:
            selected_rows.append(row)

    if args.max_rows > 0:
        selected_rows = selected_rows[: args.max_rows]

    print(f"[load] base_model    = {args.base_model}")
    print(f"[load] lora_in       = {args.lora_in}")
    print(f"[load] rollouts      = {rollouts_path}")
    print(f"[load] raw rows      = {len(raw_rows)}")
    print(f"[load] selected rows = {len(selected_rows)}")
    print(f"[train] canonicalize_valid = {canonicalize_valid}")
    print(f"[train] credit_mode = {args.credit_mode}")
    print(f"[train] objective = {args.objective}")
    if args.objective == "grpo_clip":
        print(
            f"[train] clip_ratio_low={args.clip_ratio_low} "
            f"clip_ratio_high={args.clip_ratio_high} "
            f"clip_ratio_dual={args.clip_ratio_dual} "
            f"old_logprob_mode={args.old_logprob_mode}"
        )

    if not selected_rows:
        raise RuntimeError("No usable rollout rows after advantage filtering.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[load] loading base model")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if args.full_param:
        print("[load] FULL-PARAMETER mode: training ALL base-model weights (no LoRA)")
        model = base_model
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        if not args.lora_in:
            raise SystemExit("--lora_in is required unless --full_param is set")
        print("[load] loading LoRA adapter")
        model = PeftModel.from_pretrained(
            base_model,
            args.lora_in,
            is_trainable=True,
        )

    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.to(device)

    if fsdp_enabled:
        # IMPORTANT:
        # Do NOT use size_based_auto_wrap_policy here.
        # It may wrap lm_head as a standalone FSDP module, which breaks Qwen3's final F.linear.
        # Instead, wrap only Transformer decoder blocks.
        try:
            decoder_layer_cls = {type(model.model.layers[0])}
        except Exception as exc:
            raise RuntimeError(
                "Cannot infer decoder layer class for FSDP auto-wrap. "
                "Expected model.model.layers[0] to exist for Qwen-style causal LM."
            ) from exc

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=decoder_layer_cls,
        )

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        if rank == 0:
            print(
                f"[fsdp] wrapping model: FULL_SHARD, transformer_layer_cls="
                f"{[c.__name__ for c in decoder_layer_cls]}",
                flush=True,
            )

        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )

    model.train()

    trainable, total = count_trainable_params(model)
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.4f}")

    examples = []
    skipped_build = 0
    for row in selected_rows:
        ex = build_example(
            tokenizer=tokenizer,
            row=row,
            max_seq_len=args.max_seq_len,
            canonicalize_valid=canonicalize_valid,
            credit_mode=args.credit_mode,
        )
        if ex is None:
            skipped_build += 1
            continue
        examples.append(ex)

    print(f"[build] usable examples = {len(examples)}")
    print(f"[build] skipped examples = {skipped_build}")

    if not examples:
        raise RuntimeError("No usable examples after tokenization/max_seq_len filtering.")

    old_precompute_ok = 0
    old_precompute_skipped = 0
    old_precompute_tokens = 0

    if args.objective == "grpo_clip":
        old_precompute_ok, old_precompute_skipped, old_precompute_tokens = precompute_old_log_probs(
            model=model,
            examples=examples,
            device=device,
        )
        examples = [ex for ex in examples if ex.get("old_token_log_probs") is not None]
        print(f"[old] usable examples after old_log_probs = {len(examples)}")
        if not examples:
            raise RuntimeError("No usable examples after old_log_probs precompute.")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    before = trainable_abs_sum(model)
    print(f"[probe] trainable_abs_sum_before={before:.8e}")

    update_steps = 0
    skipped_train = 0

    for epoch in range(args.epochs):
        print(f"[epoch] {epoch + 1}/{args.epochs}")
        random.shuffle(examples)

        for row_idx, ex in enumerate(examples):
            optimizer.zero_grad(set_to_none=True)

            try:
                loss, ntokens, loss_info = compute_training_loss(model, ex, device=device, args=args)
            except torch.cuda.OutOfMemoryError:
                skipped_train += 1
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"[skip][oom] row={row_idx} uid={ex['uid']} sample={ex['sample_index']}")
                continue

            if loss is None or not torch.isfinite(loss):
                skipped_train += 1
                optimizer.zero_grad(set_to_none=True)
                print(f"[skip][bad_loss] row={row_idx} uid={ex['uid']} sample={ex['sample_index']}")
                continue

            loss.backward()

            if fsdp_enabled:
                grad_norm = model.clip_grad_norm_(args.grad_clip)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )

            optimizer.step()
            update_steps += 1

            metric_suffix = ""
            if loss_info:
                metric_suffix = (
                    f" ppo_kl={loss_info.get('ppo_kl', 0.0):.6e}"
                    f" clip_hi={loss_info.get('pg_clipfrac_higher', 0.0):.4f}"
                    f" clip_lo={loss_info.get('pg_clipfrac_lower', 0.0):.4f}"
                    f" ratio_mean={loss_info.get('ratio_mean', 0.0):.6f}"
                    f" ratio_min={loss_info.get('ratio_min', 0.0):.6f}"
                    f" ratio_max={loss_info.get('ratio_max', 0.0):.6f}"
                )

            print(
                f"[train] step={update_steps} row={row_idx} "
                f"uid={ex['uid']} sample={ex['sample_index']} "
                f"loss={float(loss.detach().cpu()):.6f} "
                f"adv={ex['advantage']:.6f} reward={ex['reward']:.6f} "
                f"format={int(ex['format_ok'])} solver_score={ex['solver_score']:.6f} "
                f"tokens={ntokens} grad_norm={float(grad_norm):.8e}{metric_suffix}"
            )

    after = trainable_abs_sum(model)
    print(f"[probe] trainable_abs_sum_after={after:.8e}")
    print(f"[probe] trainable_abs_sum_delta={after - before:.8e}")
    print(f"[summary] update_steps={update_steps} skipped_train={skipped_train}")

    if update_steps <= 0:
        raise RuntimeError("No optimizer update was performed.")

    if fsdp_enabled:
        if rank == 0:
            print("[fsdp] gathering full state dict to CPU for HuggingFace save", flush=True)

        save_cfg = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=True,
        )

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_cfg):
            full_state_dict = model.state_dict()

        if rank == 0:
            model.module.save_pretrained(
                str(lora_out),
                state_dict=full_state_dict,
                safe_serialization=True,
            )
            tokenizer.save_pretrained(str(lora_out))

        del full_state_dict

        dist.barrier()
    else:
        model.save_pretrained(str(lora_out), safe_serialization=True)
        tokenizer.save_pretrained(str(lora_out))

    meta = {
        "mode": "nofsdp_full_grpo" if args.full_param else "nofsdp_lora_grpo",
        "credit_mode": args.credit_mode,
        "base_model": args.base_model,
        "lora_in": args.lora_in,
        "rollouts_jsonl": str(rollouts_path),
        "lora_out": str(lora_out),  # legacy argument name\n        "output_dir": str(lora_out),
        "lr": args.lr,
        "epochs": args.epochs,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "canonicalize_valid": canonicalize_valid,
        "raw_rows": len(raw_rows),
        "selected_rows": len(selected_rows),
        "usable_examples": len(examples),
        "update_steps": update_steps,
        "skipped_build": skipped_build,
        "skipped_train": skipped_train,
        "trainable_abs_sum_before": before,
        "trainable_abs_sum_after": after,
        "trainable_abs_sum_delta": after - before,
        "objective": args.objective,
        "objective_detail": (
            "loss = -advantage * mean_logprob(response_tokens)"
            if args.objective == "pg"
            else "PPO/GRPO clipped objective with precomputed old_log_probs"
        ),
        "clip_ratio_low": args.clip_ratio_low,
        "clip_ratio_high": args.clip_ratio_high,
        "clip_ratio_dual": args.clip_ratio_dual,
        "old_logprob_mode": args.old_logprob_mode,
        "old_precompute_ok": old_precompute_ok,
        "old_precompute_skipped": old_precompute_skipped,
        "old_precompute_tokens": old_precompute_tokens,
    }

    meta_path = lora_out / "rzero_fsdp_grpo_meta.json" if fsdp_enabled else lora_out / "rzero_nofsdp_lora_grpo_meta.json"

    if rank == 0:
        meta["mode"] = "fsdp_full_grpo" if fsdp_enabled and args.full_param else meta["mode"]
        meta["fsdp_enabled"] = bool(fsdp_enabled)
        meta["world_size"] = int(world_size)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[save] output_dir = {lora_out}")
        print(f"[save] meta     = {meta_path}")

    if fsdp_enabled:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
