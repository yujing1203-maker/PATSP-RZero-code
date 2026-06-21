#!/usr/bin/env bash
# Full-parameter (qwen3-4b-base) PATSP-vs-R-Zero self-play for the benchmarks that lack a native
# co-evolution run, run SEQUENTIALLY on one node's 2 GPUs (R-Zero arm GPU0, PATSP-outcome arm GPU1).
# Generic over --env (no HeroBench server/port/db). Round-level RESUME + keep-best disk-hygiene are in
# the driver, so an interrupted run (e.g. interactive-node reassignment) is recoverable by re-running
# this script (each env's driver resumes from its rounds.json + last/best checkpoint).
#
# Default envs: scienceworld then alfworld. Config mirrors the HeroBench full-param middle-lr run.
set -uo pipefail
cd "$(dirname "$0")/../../.."
PROJ="$PWD"
export RZERO_STORAGE="${RZERO_STORAGE:-$HOME/RZero_storage}"
export RZERO_PROJECT="$PROJ"
export PYTHONPATH="$PROJ/scripts/rzero_nofsdp_lora"
export BASE_MODEL="${BASE_MODEL:-$PROJ/data/models/qwen3-4b-base}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$RZERO_STORAGE/alfworld_data}"
# shared training/rollout knobs (full-param middle lr; weak base -> accept all train tasks)
export RZERO_TRAIN_LR_FULL="${RZERO_TRAIN_LR_FULL:-2e-6}"
export RZERO_TRAIN_EPOCHS="${RZERO_TRAIN_EPOCHS:-1}"
export RZERO_STAGE_MAX_MODEL_LEN="${RZERO_STAGE_MAX_MODEL_LEN:-8192}"
export RZERO_GEN_NUM_TASKS="${RZERO_GEN_NUM_TASKS:-9}"
export PATSP_MAX_STEPS="${PATSP_MAX_STEPS:-20}"
export PATSP_MAX_REPLANS="${PATSP_MAX_REPLANS:-3}"
export RZERO_SCORE_MIN="${RZERO_SCORE_MIN:-0.0}"
export LH_EVAL_GROUP="${LH_EVAL_GROUP:-3}"
export RZERO_MAX_SECONDS="${RZERO_MAX_SECONDS:-0}"   # interactive: no walltime self-stop (120h alloc)
# qwen3-4b-base degenerates into single-token repetition on ScienceWorld prompts (all-zero rewards);
# the repo's existing guard breaks the loop. Applied globally (harmless on envs that don't need it).
export RZERO_REPETITION_PENALTY="${RZERO_REPETITION_PENALTY:-1.2}"
ROUNDS="${ROUNDS:-6}"
DRIVER="scripts/rzero_nofsdp_lora/patsp/patsp_selfplay.py"

run_env() {  # env  d2_tasks  exp  d0
  local ENV="$1" D2="$2" EXP="$3" D0="${4:-0.5}"
  if [ ! -f "$D2" ]; then echo "### SKIP $ENV: missing D2 $D2"; return 0; fi
  mkdir -p "$EXP"
  echo "### SELFPLAY $ENV START $(date) exp=$EXP rounds=$ROUNDS d0=$D0 lr_full=$RZERO_TRAIN_LR_FULL rep_pen=$RZERO_REPETITION_PENALTY"
  CUDA_VISIBLE_DEVICES=0 python "$DRIVER" --name rzero --arm rzero --env "$ENV" --rounds "$ROUNDS" \
    --d0 "$D0" --step 0.0 --d2_tasks "$D2" --seed 1 --exp "$EXP" --full_param 1 \
    > "$EXP/rzero.log" 2>&1 &
  local PRZ=$!
  CUDA_VISIBLE_DEVICES=1 python "$DRIVER" --name patsp_oc --arm patsp --env "$ENV" --rounds "$ROUNDS" \
    --d0 "$D0" --step 0.0 --d2_tasks "$D2" --seed 1 --exp "$EXP" --full_param 1 \
    --credit_mode outcome --outcome_lambda 0.5 \
    > "$EXP/patsp_oc.log" 2>&1 &
  local PPA=$!
  wait $PRZ; local RZ=$?
  wait $PPA; local PA=$?
  echo "### SELFPLAY $ENV DONE $(date) rzero_rc=$RZ patsp_rc=$PA"
  echo "###   rzero rounds:";  cat "$EXP/rzero/rounds.json"    2>/dev/null | python -c "import sys,json;print([(r['round'],r['D2']) for r in json.load(sys.stdin)])" 2>/dev/null || echo "(none)"
  echo "###   patsp rounds:";  cat "$EXP/patsp_oc/rounds.json" 2>/dev/null | python -c "import sys,json;print([(r['round'],r['D2']) for r in json.load(sys.stdin)])" 2>/dev/null || echo "(none)"
}

# ALFWorld first: 4B-instruct probe = {1.0:7, 0.5:1} on 8 episodes (strong signal, clean actions).
run_env alfworld     "$RZERO_STORAGE/patsp_eval/alfworld_D2.jsonl"          "$RZERO_STORAGE/patsp_alfworld_full"     0.5
# ScienceWorld DISABLED (2026-06-10 probes): qwen3-4b-base emits degenerate/garbage text; the
# 4B-INSTRUCT fixes the format (clean <action>focus on ...</action>) but still scores 0/8 even on the
# easiest family (find-living-thing) at 12 AND 24 steps — it focuses a wrong object, which ScienceWorld
# scores 0 terminally. ScienceWorld co-evolution needs the 7B-instruct (validated easy=0.52).
# run_env scienceworld "$RZERO_STORAGE/patsp_eval/scienceworld_D2_easy.jsonl" "$RZERO_STORAGE/patsp_scienceworld_full" 0.0
echo "### SEQ_ALL_DONE $(date)"
