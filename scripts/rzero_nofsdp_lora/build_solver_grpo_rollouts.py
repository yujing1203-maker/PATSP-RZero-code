# -*- coding: utf-8 -*-
"""
Build Solver GRPO rollouts from parquet training data.

Input parquet schema:
  problem, answer, source, verification, note

For each problem:
  S_t samples group_size solutions.
  examples.reward_function.math.compute_score is used when possible.
  group-wise GRPO advantage is computed from rewards.

No FSDP.
No training here.
"""

import argparse
import inspect
import json
import math
import os
import sys
from collections import defaultdict, Counter
from pathlib import Path

import vllm
from datasets import Dataset
from transformers import AutoTokenizer
from mathruler.grader import extract_boxed_content, grade_answer

# Robust import for the new verifiers subpackage.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from examples.reward_function.math import compute_score as original_math_compute_score
except Exception as exc:
    original_math_compute_score = None
    ORIGINAL_IMPORT_ERROR = repr(exc)
else:
    ORIGINAL_IMPORT_ERROR = ""

try:
    from vllm.lora.request import LoRARequest
except Exception as exc:
    raise RuntimeError("vLLM LoRARequest is unavailable. Please check vLLM installation.") from exc

# The verifiers subpackage is only required for --multistep. Import lazily-guarded so
# the default (non-multistep) GSM8K pipeline stays runnable even if it is absent.
try:
    from verifiers.base import compute_stage_potentials, sigma_key
    from verifiers.math_multistep import MathMultiStepVerifier
    VERIFIERS_IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001
    compute_stage_potentials = None
    sigma_key = None
    MathMultiStepVerifier = None
    VERIFIERS_IMPORT_ERROR = repr(exc)


SOLVER_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

# Multi-step solver prompt: force each intermediate calculation onto its own line
# as `Step k: <reasoning> = \boxed{<value>}`, then the final \boxed{}.
# This makes intermediate values explicitly boxed -> directly gradable by the verifier.
SOLVER_MULTISTEP_SYSTEM_PROMPT = (
    "Solve the problem step by step. Present EACH intermediate calculation on its own "
    "line as `Step k: <reasoning> = \\boxed{<value>}`, then give the final answer as "
    "`\\boxed{<final>}`. Free-form reasoning between steps is allowed."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--solver_lora", required=True)
    parser.add_argument("--train_parquet", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--enforce_eager", action="store_true")

    parser.add_argument(
        "--multistep",
        action="store_true",
        help=(
            "Multi-checkpoint mode: use the multi-step solver "
            "prompt, read per-problem gold checkpoints from the train parquet `checkpoints` "
            "column, attach MathMultiStepVerifier, and emit stage_records/stage_potentials/"
            "outcome_advantage/sigma_keys/stage_advantages. Off by default; when off the "
            "existing GSM8K rollout behavior is byte-identical."
        ),
    )
    parser.add_argument(
        "--stage_alpha",
        type=float,
        default=1.0,
        help="alpha weight for milestone term m_i in compute_stage_potentials (--multistep).",
    )
    parser.add_argument(
        "--stage_beta",
        type=float,
        default=1.0,
        help="beta weight for recovery term r_i in compute_stage_potentials (--multistep).",
    )
    parser.add_argument(
        "--stage_gamma",
        type=float,
        default=0.1,
        help="gamma weight for normalized cost term c_i in compute_stage_potentials (--multistep).",
    )
    return parser.parse_args()


def build_solver_prompt(tokenizer, problem: str, system_prompt: str = SOLVER_SYSTEM_PROMPT) -> str:
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"system: {system_prompt}\nuser: {problem}\nassistant:"


def parse_checkpoints(raw) -> list:
    """Parse the parquet `checkpoints` column (JSON string of ordered checkpoints).

    Expected element shape: list[{id:str, gold:str, order:int, source:str}]. Tolerates an
    already-decoded list (datasets sometimes returns native python objects). Returns
    a list ordered by `order` when present; empty list on any failure.
    """
    if raw is None:
        return []

    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            raw = json.loads(s)
        except Exception:
            return []

    if not isinstance(raw, (list, tuple)):
        return []

    checkpoints = [dict(c) for c in raw if isinstance(c, dict)]

    def _order_key(c):
        try:
            return int(c.get("order", 0))
        except Exception:
            return 0

    checkpoints.sort(key=_order_key)
    return checkpoints


def build_gold(src: dict, checkpoints: list) -> dict:
    """gold dict consumed by the verifier: {answer, checkpoints, ...}."""
    return {
        "answer": src.get("answer", ""),
        "checkpoints": checkpoints,
        "problem": src.get("problem", ""),
        "source": src.get("source", ""),
    }


def stage_costs_from_records(stage_records: list, completion: str) -> list:
    """Normalized per-stage cost c_i = (chars in this stage's span) / (total span chars).

    Used as `cost_per_stage` for compute_stage_potentials. char_span indexes into
    `completion`. Returns a list aligned to stage_records, or
    None when no positive-length spans exist (verifier may then use its own default).
    """
    lengths = []
    for rec in stage_records:
        span = rec.char_span if hasattr(rec, "char_span") else (rec.get("char_span") if isinstance(rec, dict) else None)
        if span and len(span) == 2:
            try:
                length = max(0, int(span[1]) - int(span[0]))
            except Exception:
                length = 0
        else:
            length = 0
        lengths.append(length)

    total = sum(lengths)
    if total <= 0:
        return None

    return [length / total for length in lengths]


def fallback_math_reward(completion: str, answer: str) -> float:
    pred = extract_boxed_content(completion)
    if not pred:
        return 0.0

    try:
        ok = grade_answer(pred, answer) or grade_answer(answer, pred)
    except Exception:
        ok = pred.strip() == str(answer).strip()

    return 1.0 if ok else 0.0


def normalize_reward_value(x):
    if isinstance(x, bool):
        return 1.0 if x else 0.0

    if isinstance(x, (int, float)):
        return float(x)

    if isinstance(x, dict):
        for key in ["score", "overall", "reward", "accuracy"]:
            if key in x:
                try:
                    return float(x[key])
                except Exception:
                    pass

    return None


def compute_reward_with_original_math(completion: str, answer: str) -> float:
    """
    Prefer original examples.reward_function.math.compute_score.

    In this repository the original signature is usually:

        compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1)

    So the first attempt must pass one-element lists.
    Fallback is only used if the original reward cannot be called.
    """
    if original_math_compute_score is None:
        return fallback_math_reward(completion, answer)

    attempts = [
        lambda: original_math_compute_score([completion], [answer]),
        lambda: original_math_compute_score(predicts=[completion], ground_truths=[answer]),
        lambda: original_math_compute_score([completion], [answer], format_weight=0.1),
        lambda: original_math_compute_score(solution_str=completion, ground_truth=answer),
        lambda: original_math_compute_score(completion, answer),
    ]

    for fn in attempts:
        try:
            value = fn()

            if isinstance(value, list) and value:
                reward = normalize_reward_value(value[0])
            else:
                reward = normalize_reward_value(value)

            if reward is not None:
                return float(reward)

        except TypeError:
            continue
        except Exception:
            continue

    return fallback_math_reward(completion, answer)


def compute_group_advantages(rows):
    by_uid = defaultdict(list)
    for idx, r in enumerate(rows):
        by_uid[r["uid"]].append(idx)

    advantages = [0.0 for _ in rows]

    for uid, indices in by_uid.items():
        rewards = [float(rows[i]["reward"]) for i in indices]

        if len(rewards) <= 1:
            continue

        mean = sum(rewards) / len(rewards)
        var_num = sum((x - mean) ** 2 for x in rewards)
        std = math.sqrt(var_num / (len(rewards) - 1)) if len(rewards) > 1 else 0.0

        if std < 1e-8:
            continue

        for i in indices:
            advantages[i] = (float(rows[i]["reward"]) - mean) / (std + 1e-6)

    return advantages


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rollout_path = out_dir / "solver_grpo_rollouts.jsonl"
    summary_path = out_dir / "summary.json"

    verifier = None
    if args.multistep:
        if MathMultiStepVerifier is None or compute_stage_potentials is None or sigma_key is None:
            raise RuntimeError(
                "--multistep requires the verifiers subpackage "
                "(verifiers.base + verifiers.math_multistep) but it failed to import: "
                f"{VERIFIERS_IMPORT_ERROR}"
            )
        verifier = MathMultiStepVerifier()
        print("[multistep] enabled: attaching MathMultiStepVerifier")
        print(f"[multistep] solver prompt = multi-step boxed-checkpoint prompt")

    ds = Dataset.from_parquet(args.train_parquet)
    train_rows = [dict(x) for x in ds]

    print(f"[load] train rows  = {len(train_rows)}")
    print(f"[load] train file  = {args.train_parquet}")
    print(f"[load] base_model  = {args.base_model}")
    print(f"[load] solver_lora = {args.solver_lora}")
    print(f"[align] reward = examples.reward_function.math.compute_score")
    if original_math_compute_score is None:
        print(f"[warn] original math reward import failed: {ORIGINAL_IMPORT_ERROR}")
        print("[warn] fallback reward = mathruler boxed answer grading")
    else:
        try:
            print(f"[align] math.compute_score signature = {inspect.signature(original_math_compute_score)}")
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    llm = vllm.LLM(
        model=args.base_model,
        tokenizer=args.base_model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=int(os.environ.get("RZERO_VLLM_TP", "1")),  # dual-card TP via RZERO_VLLM_TP
        enable_lora=True,
        max_lora_rank=args.lora_rank,
        max_loras=4,
        max_cpu_loras=8,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )

    sampling_kwargs = dict(
        n=args.group_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    if args.top_k is not None and args.top_k > 0:
        sampling_kwargs["top_k"] = args.top_k

    if tokenizer.eos_token_id is not None:
        sampling_kwargs["stop_token_ids"] = [tokenizer.eos_token_id]

    sampling_params = vllm.SamplingParams(**sampling_kwargs)

    solver_system_prompt = SOLVER_MULTISTEP_SYSTEM_PROMPT if args.multistep else SOLVER_SYSTEM_PROMPT
    prompts = [
        build_solver_prompt(tokenizer, r["problem"], system_prompt=solver_system_prompt)
        for r in train_rows
    ]

    lora_request = LoRARequest(
        lora_name="solver_rollout",
        lora_int_id=31,
        lora_path=args.solver_lora,
    )

    print("[vllm] start solver rollout generation")
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=lora_request,
        use_tqdm=True,
    )
    print("[vllm] finish solver rollout generation")

    rollout_rows = []

    # Multistep verifier bookkeeping. Aggregated only when
    # --multistep is on; left empty otherwise so the default path is unchanged.
    total_milestones_recalled = 0
    total_gold_checkpoints = 0
    rows_with_records = 0
    checkpoints_missing_problems = 0

    for uid, (src, output) in enumerate(zip(train_rows, outputs)):
        problem = src["problem"]
        answer = src["answer"]
        prompt = prompts[uid]

        gold = None
        if args.multistep:
            checkpoints = parse_checkpoints(src.get("checkpoints"))
            if not checkpoints:
                checkpoints_missing_problems += 1
            gold = build_gold(src, checkpoints)
            total_gold_checkpoints += len(checkpoints) * len(output.outputs)

        for sample_index, one in enumerate(output.outputs):
            completion = one.text or ""
            pred = extract_boxed_content(completion)
            reward = compute_reward_with_original_math(completion, answer)

            row = {
                "uid": uid,
                "sample_index": sample_index,
                "prompt": prompt,
                "completion": completion,
                "completion_for_train": completion,
                "problem": problem,
                "answer": answer,
                "pred_answer": pred,
                "reward": float(reward),
                "source": src.get("source", ""),
                "verification": src.get("verification", ""),
                "note": src.get("note", ""),
                "solver_lora": args.solver_lora,
            }

            if args.multistep:
                stage_records = verifier.verify(completion, gold)

                # NOTE on the recovery term r_i: on this single-record-per-checkpoint
                # MATH substrate the recovery term is identically 0 -- recovery requires
                # a predicate to go sat -> unsat -> sat across multiple records for the
                # same checkpoint, which only arises in a multi-turn setting (not
                # exercised here). compute_stage_potentials still accepts stage_beta and
                # computes r_i, but here every r_i == 0, so stage_beta has no effect on
                # the emitted potentials. Documented only; logic is intentionally unchanged.
                potentials = compute_stage_potentials(
                    stage_records,
                    alpha=args.stage_alpha,
                    beta=args.stage_beta,
                    gamma=args.stage_gamma,
                    cost_per_stage=stage_costs_from_records(stage_records, completion),
                )

                # sigma at each stage boundary: state of satisfied predicates ENTERING
                # stage i, i.e. computed from records strictly before i.
                stage_keys = [sigma_key(stage_records[:i]) for i in range(len(stage_records))]

                row["stage_records"] = [rec.to_dict() for rec in stage_records]
                row["stage_potentials"] = [float(p) for p in potentials]
                row["sigma_keys"] = list(stage_keys)
                # outcome_advantage filled below (group-relative GRPO on terminal reward).
                row["outcome_advantage"] = 0.0
                # stage_advantages stays null until apply_credit_head.py fills it.
                row["stage_advantages"] = None

                if stage_records:
                    rows_with_records += 1
                    total_milestones_recalled += sum(
                        1 for rec in stage_records if rec.after == "sat"
                    )

            rollout_rows.append(row)

    advantages = compute_group_advantages(rollout_rows)
    for r, adv in zip(rollout_rows, advantages):
        r["advantage"] = float(adv)

    # outcome_advantage = the existing group-relative GRPO advantage on terminal reward.
    # In uniform mode this equals `advantage`; we compute it from the
    # same terminal `reward` so the two stay identical and the field is explicit.
    if args.multistep:
        outcome_advantages = compute_group_advantages(rollout_rows)
        for r, oadv in zip(rollout_rows, outcome_advantages):
            r["outcome_advantage"] = float(oadv)

    with rollout_path.open("w", encoding="utf-8") as f:
        for r in rollout_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    groups = []
    by_uid = defaultdict(list)
    for r in rollout_rows:
        by_uid[r["uid"]].append(r)

    for uid in sorted(by_uid):
        group = sorted(by_uid[uid], key=lambda x: x["sample_index"])
        rewards = [float(x["reward"]) for x in group]
        advs = [float(x["advantage"]) for x in group]
        groups.append(
            {
                "uid": uid,
                "rewards": rewards,
                "advantages": advs,
                "nonzero_advantage": any(abs(a) > 1e-8 for a in advs),
            }
        )
        print(f"[group] uid={uid} rewards={rewards} advantages={advs}")

    summary = {
        "backend": "vllm",
        "base_model": args.base_model,
        "solver_lora": args.solver_lora,
        "train_parquet": args.train_parquet,
        "num_prompts": len(train_rows),
        "group_size": args.group_size,
        "num_rollouts": len(rollout_rows),
        "reward_function": "examples.reward_function.math.compute_score",
        "fallback_reward_used_if_import_failed": original_math_compute_score is None,
        "reward_distribution": dict(Counter(float(r["reward"]) for r in rollout_rows)),
        "nonzero_advantage_rows": sum(abs(float(r["advantage"])) > 1e-8 for r in rollout_rows),
        "nonzero_advantage_groups": sum(bool(g["nonzero_advantage"]) for g in groups),
        "groups": groups,
        "rollout_path": str(rollout_path),
    }

    if args.multistep:
        milestone_recall = (
            total_milestones_recalled / total_gold_checkpoints
            if total_gold_checkpoints > 0
            else 0.0
        )
        summary["multistep"] = {
            "enabled": True,
            "solver_system_prompt": SOLVER_MULTISTEP_SYSTEM_PROMPT,
            "verifier": "verifiers.math_multistep.MathMultiStepVerifier",
            "stage_alpha": args.stage_alpha,
            "stage_beta": args.stage_beta,
            "stage_gamma": args.stage_gamma,
            "rows_with_stage_records": rows_with_records,
            "total_gold_checkpoints": total_gold_checkpoints,
            "total_milestones_recalled": total_milestones_recalled,
            # Honest known limitation: expect ~0.6 due to alternate valid solution paths.
            "milestone_recall": milestone_recall,
            "problems_missing_checkpoints": checkpoints_missing_problems,
            "stage_advantages_status": "null (filled by apply_credit_head.py)",
            # The recovery term r_i in compute_stage_potentials is identically 0 on
            # this single-record-per-checkpoint MATH substrate; recovery is a
            # multi-turn feature, not exercised here. stage_beta is therefore inert
            # here (kept for forward-compatibility with the multi-turn substrate).
            "recovery_term_active": False,
            "recovery_term_note": (
                "r_i == 0 on single-record-per-checkpoint MATH; recovery is a "
                "multi-turn feature, not exercised here. stage_beta has no effect here."
            ),
        }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print(f"[save] rollouts = {rollout_path}")
    print(f"[save] summary  = {summary_path}")
    print(f"[summary] nonzero_advantage_groups={summary['nonzero_advantage_groups']}/{len(groups)}")
    if args.multistep:
        print(
            f"[multistep] milestone_recall={summary['multistep']['milestone_recall']:.4f} "
            f"({total_milestones_recalled}/{total_gold_checkpoints}) "
            f"rows_with_stage_records={rows_with_records} "
            f"problems_missing_checkpoints={checkpoints_missing_problems}"
        )


if __name__ == "__main__":
    main()
