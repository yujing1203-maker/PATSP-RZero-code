#!/usr/bin/env bash
# One-click R-Zero vs PATSP co-evolution on ONE benchmark.
# Serial mode: R-Zero runs first on GPU1, then PATSP runs on GPU2.
# Full-parameter by default with the validated stable config.
# Full-parameter by default with the validated stable config (4B-instruct, lr=2e-6).
# The driver has round-level RESUME + keep-best checkpoints: re-running this script after an
# interruption continues from the last completed round.
#
# Usage:  bash scripts/run/selfplay.sh ENV [options]
#   ENV            textcraft | scienceworld | alfworld | herobench
#   --rounds N     co-evolution rounds        (default 6)
#   --model PATH   full base checkpoint       (default data/models/qwen3-4b-instruct;
#                                              with --lora defaults to data/models/qwen3-4b-instruct
#                                              = the base the patsp_init/ adapters were built on)
#   --lr LR        learning rate              (full-param default 2e-6; LoRA default 3e-5)
#   --d0 D         training difficulty        (default per-env: sciworld 0.0, others 0.5)
#   --exp DIR      experiment dir             (default $RZERO_STORAGE/patsp_<env>_full)
#   --lora         LoRA mode instead of full-param (cold-starts from $RZERO_STORAGE/patsp_init/)
# Examples:
#   bash scripts/run/selfplay.sh alfworld
#   bash scripts/run/selfplay.sh herobench --rounds 6
#   bash scripts/run/selfplay.sh textcraft --lora          # auto-uses the 4B base of the adapters
#   RZERO_MAX_SECONDS=14400 bash scripts/run/selfplay.sh textcraft   # clean self-stop after ~4h
#
# NOTE: run a PROBE first (scripts/run/probe.sh); for multi-hour runs set
#       RZERO_MAX_SECONDS (clean self-stop + per-round resume via rounds.json) and wrap this
#       in your own cluster scheduler (PBS/SLURM) rather than an interactive shell/tmux.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$PROJ"   # driver launches subprocesses with repo-relative paths -> must run from repo root
require_env_arg "${1:-}" || exit 1
ENV="$1"; shift
need(){ [ -n "${2:-}" ] || { echo "ERROR: missing value for $1" >&2; exit 1; }; }
ROUNDS=6; MODEL=""; LR=""; D0="$(d0_for "$ENV")"; EXP=""; FULLP=1
while [ $# -gt 0 ]; do case "$1" in
  --rounds) need "$1" "${2:-}"; ROUNDS="$2"; shift 2 ;;
  --model)  need "$1" "${2:-}"; MODEL="$2";  shift 2 ;;
  --lr)     need "$1" "${2:-}"; LR="$2";     shift 2 ;;
  --d0)     need "$1" "${2:-}"; D0="$2";     shift 2 ;;
  --exp)    need "$1" "${2:-}"; EXP="$2";    shift 2 ;;
  --lora)   FULLP=0;            shift ;;
  *) echo "unknown option: $1" >&2; exit 1 ;;
esac; done

# Default model & lr depend on mode. LoRA cold-starts from the FIXED patsp_init/ adapters, whose
# base is qwen2.5-7b-instruct -> applying them to a 4B engine is a shape-mismatch crash.
USER_MODEL=1; [ -z "$MODEL" ] && USER_MODEL=0
if [ "$FULLP" = "1" ]; then
  MODEL="${MODEL:-$DEFAULT_MODEL}"; LR="${LR:-2e-6}"
  export RZERO_TRAIN_LR_FULL="$LR"
else
  MODEL="${MODEL:-$DEFAULT_MODEL}"; LR="${LR:-3e-5}"
  export RZERO_TRAIN_LR="$LR"
  # sanity: warn if base doesn't match the init adapters' base_model_name_or_path (qwen3-4b-instruct)
  ADP="$RZERO_STORAGE/patsp_init/planner/adapter_config.json"
  if [ -f "$ADP" ]; then
    WANT="$(python -c "import json;print(json.load(open('$ADP')).get('base_model_name_or_path',''))" 2>/dev/null)"
    [ -n "$WANT" ] && [ "$(basename "$WANT")" != "$(basename "$MODEL")" ] && \
      echo "WARN: --lora adapters were built on '$(basename "$WANT")' but --model is '$(basename "$MODEL")' — likely a shape mismatch." >&2
  fi
fi
[ -f "$MODEL/config.json" ] || { echo "ERROR: model dir invalid: $MODEL" >&2; exit 1; }
[ "$ENV" = "scienceworld" ] && echo "WARN: ScienceWorld produced ZERO signal at 4B (even instruct); needs a 7B-instruct. Probe first!" >&2

# HeroBench has its own launcher (per-port server+DB pip install isolation, train/eval task split) — delegate.
# Forward BASE_MODEL only when the user explicitly chose one, so the launcher keeps its correct
# per-mode default otherwise. Forward d0/lr through dedicated knobs.
if [ "$ENV" = "herobench" ]; then
  exec env FULL_PARAM="$FULLP" ROUNDS="$ROUNDS" HB_D0="$D0" \
       $([ "$USER_MODEL" = "1" ] && echo BASE_MODEL="$MODEL") \
       $([ "$FULLP" = "1" ] && echo RZERO_TRAIN_LR_FULL="$LR" || echo RZERO_TRAIN_LR="$LR") \
       ${EXP:+EXP="$EXP"} \
       bash "$PROJ/scripts/rzero_nofsdp_lora/patsp/run_herobench_selfplay.sh"
fi

# 8 * 24GB (vllm checkpoint, 4b -> n -> merge)

EXP="${EXP:-$RZERO_STORAGE/patsp_${ENV}_full}"
D2="$(d2_for "$ENV")"; STEPS="$(steps_for "$ENV")"
export BASE_MODEL="$MODEL"
export PATSP_MAX_STEPS="$STEPS" RZERO_GEN_NUM_TASKS="${RZERO_GEN_NUM_TASKS:-9}" RZERO_SCORE_MIN="${RZERO_SCORE_MIN:-0.0}"
export LH_EVAL_GROUP="${LH_EVAL_GROUP:-3}" RZERO_MAX_SECONDS="${RZERO_MAX_SECONDS:-0}"
mkdir -p "$EXP"
DRIVER="$PROJ/scripts/rzero_nofsdp_lora/patsp/patsp_selfplay.py"
echo "== selfplay $ENV | full_param=$FULLP rounds=$ROUNDS lr=$LR d0=$D0 model=$(basename "$MODEL") exp=$EXP =="

echo "  R-Zero arm  GPU1 -> $EXP/rzero.log"
CUDA_VISIBLE_DEVICES=1 python "$DRIVER" --name rzero --arm rzero --env "$ENV" --rounds "$ROUNDS" \
  --d0 "$D0" --step 0.0 --d2_tasks "$D2" --seed 1 --exp "$EXP" --full_param "$FULLP" \
  > "$EXP/rzero.log" 2>&1
R1=$?

if [ "$R1" -ne 0 ]; then
  echo "== R-Zero FAILED rc=$R1 =="
  echo "-- tail of $EXP/rzero.log --"
  tail -120 "$EXP/rzero.log"
  exit "$R1"
fi

echo "  PATSP arm   GPU2 -> $EXP/patsp_oc.log"
CUDA_VISIBLE_DEVICES=2 python "$DRIVER" --name patsp_oc --arm patsp --env "$ENV" --rounds "$ROUNDS" \
  --d0 "$D0" --step 0.0 --d2_tasks "$D2" --seed 1 --exp "$EXP" --full_param "$FULLP" \
  --credit_mode outcome --outcome_lambda 0.5 \
  > "$EXP/patsp_oc.log" 2>&1
R2=$?

if [ "$R2" -ne 0 ]; then
  echo "== PATSP FAILED rc=$R2 =="
  echo "-- tail of $EXP/patsp_oc.log --"
  tail -120 "$EXP/patsp_oc.log"
  exit "$R2"
fi

echo "== DONE rzero_rc=$R1 patsp_rc=$R2 =="

for a in rzero patsp_oc; do
  echo "-- $a rounds:"
  python -c "import json;print([(r['round'],r['D2']) for r in json.load(open('$EXP/$a/rounds.json'))])" 2>/dev/null || echo "(none)"
done

RC=0
[ "$R1" -ne 0 ] && RC=$R1
[ "$R2" -ne 0 ] && RC=$R2
exit $RC