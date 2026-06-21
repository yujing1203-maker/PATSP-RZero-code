#!/usr/bin/env bash
# One-click checkpoint evaluation on a frozen D2 set. ALWAYS evaluate `base` alongside a
# trained checkpoint — a number without its base reference is meaningless.
#
# Usage:
#   bash scripts/run/eval.sh ENV base                              [options]   # base model
#   bash scripts/run/eval.sh ENV --planner DIR --executor DIR      [options]   # full-param ckpts
#   bash scripts/run/eval.sh ENV --planner-lora DIR --executor-lora DIR [opts] # LoRA adapters
# Options:
#   --model PATH   base model (default data/models/qwen3-4b-instruct; MUST match training base)
#   --group N      episodes per task (default 8 — bigger than training eval to cut noise)
#   --gpu G        CUDA device (default 0)
#   --big          textcraft only: use the 30-task EVAL30 set instead of the 10-task D2
# Examples:
#   bash scripts/run/eval.sh herobench base
#   bash scripts/run/eval.sh herobench --planner $RZERO_STORAGE/patsp_herobench_full_lr2e6/patsp_oc/r2_planner_model \
#                                      --executor $RZERO_STORAGE/patsp_herobench_full_lr2e6/patsp_oc/r2_exec_model
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_env_arg "${1:-}" || exit 1
ENV="$1"; shift
MODEL="$DEFAULT_MODEL"; GROUP=8; GPU=0; BIG=0; PM=""; EM=""; PL=""; EL=""; TAG="ckpt"
need(){ [ -n "${2:-}" ] || { echo "ERROR: missing value for $1" >&2; exit 1; }; }
while [ $# -gt 0 ]; do case "$1" in
  base) TAG="base"; shift ;;
  --planner) need "$1" "${2:-}"; PM="$2"; shift 2 ;; --executor) need "$1" "${2:-}"; EM="$2"; shift 2 ;;
  --planner-lora) need "$1" "${2:-}"; PL="$2"; shift 2 ;; --executor-lora) need "$1" "${2:-}"; EL="$2"; shift 2 ;;
  --model) need "$1" "${2:-}"; MODEL="$2"; shift 2 ;; --group) need "$1" "${2:-}"; GROUP="$2"; shift 2 ;;
  --gpu) need "$1" "${2:-}"; GPU="$2"; shift 2 ;; --big) BIG=1; shift ;;
  *) echo "unknown option: $1" >&2; exit 1 ;;
esac; done
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "ERROR: --gpu must be a single index (got '$GPU')" >&2; exit 1; }
D2="$(d2_for "$ENV")"; [ "$ENV" = "textcraft" ] && [ "$BIG" = "1" ] && D2="$(d2_for textcraft30)"
STEPS="$(steps_for "$ENV")"
OUT="/tmp/eval_${ENV}_${TAG}_$$"; mkdir -p "$OUT"
export HEROBENCH_PORT=$((8300 + GPU))   # set unconditionally; never inherit a run's port

ARGS=(--env "$ENV" --tasks_jsonl "$D2" --num_tasks 0 --out_jsonl "$OUT/roll.jsonl"
      --group_size "$GROUP" --max_steps "$STEPS" --max_replans 3
      --policy vllm --base_model "$MODEL"
      --temperature 0.7 --top_p 0.95 --max_tokens 256 --gpu_memory_utilization "${RZERO_EVAL_GPU_MEMORY_UTILIZATION:-0.40}")
if [ "$TAG" = "base" ]; then
  # Base eval already uses the shared --base_model engine in ARGS.
  # Do NOT append --planner_model/--executor_model here, or vLLM loads
  # two identical full-model engines. This is especially important for
  # LongReason 16k evaluation.
  :
elif [ -n "$PM" ] && [ -n "$EM" ]; then
  ARGS+=(--planner_model "$PM" --executor_model "$EM")
elif [ -n "$PL" ] && [ -n "$EL" ]; then
  ARGS+=(--planner_lora "$PL" --executor_lora "$EL")
else
  echo "ERROR: give 'base', or --planner+--executor (full ckpts), or --planner-lora+--executor-lora" >&2; exit 1
fi

echo "== eval $ENV [$TAG] | $(basename "$D2") x group$GROUP | gpu=$GPU =="
CUDA_VISIBLE_DEVICES="$GPU" python "$PROJ/scripts/rzero_nofsdp_lora/build_agentic_rollouts.py" "${ARGS[@]}" \
  > "$OUT/log" 2>&1 || { echo "EVAL FAILED — tail of $OUT/log:"; tail -8 "$OUT/log"; exit 1; }
M="$(mean_from_summary "$OUT")"
[ -n "$M" ] || { echo "ERROR: could not parse mean_terminal_reward from $OUT/summary.json" >&2; tail -8 "$OUT/log"; exit 1; }
N="$(wc -l < "$OUT/roll.jsonl" 2>/dev/null)"; N="${N:-0}"
echo "mean_terminal_reward = $M   (n=$N episodes; artifacts: $OUT)"
echo "(remember: compare against 'bash scripts/run/eval.sh $ENV base' — never read a number alone)"
