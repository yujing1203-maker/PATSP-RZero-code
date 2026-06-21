#!/usr/bin/env bash
# PATSP on HeroBench (benchmark #4): R-Zero (sparse final reward) vs PATSP-outcome
# (outcome-grounded dense credit) self-play, ONE arm per GPU, fully isolated.
#
# Each arm runs patsp_selfplay.py end-to-end: gen learnable tasks -> rollout group
# trajectories -> [PATSP arm] rewrite reward via outcome_credit -> split-by-role ->
# GRPO train planner'+executor' (lr=3e-5/1ep, the STABLE config) -> eval on the FROZEN
# held-out D2 with RAW terminal_reward (the true metric). Same tasks/trainer for both
# arms -> a D2 gap isolates the trajectory-credit contribution. Single seed, GPU-only.
#
# Isolation: each arm gets its OWN HeroBench server port + SQLite DB (artifact_<port>.db),
# so the two arms never share or rm_db-wipe one another's world. The held-out eval tasks
# (HEROBENCH_TRAIN_EXCLUDE) are excluded from the challenger's training-task sampling, so
# D2 stays disjoint from what each solver trains on (HeroBench's pool is only 13 tasks).
#
# For long runs use RZERO_MAX_SECONDS self-stop + a PBS/SLURM resubmit script (not interactive
# tmux, which dies on disconnect). See repo-root TUTORIAL.md.
set -euo pipefail
cd "$(dirname "$0")/../../.."                      # -> project root
PROJ="$PWD"

export RZERO_STORAGE="${RZERO_STORAGE:-$HOME/RZero_storage}"
export RZERO_PROJECT="$PROJ"
export PYTHONPATH="$PROJ/scripts/rzero_nofsdp_lora"
# Blackwell sm_120: bundled FlashAttention + native sampler (system gcc 8.5 < 9 => flashinfer JIT fails)
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# FULL_PARAM=1 -> full-parameter fine-tune (no LoRA): each role its own full checkpoint + own vLLM
# engine (driver halves gpu_util to fit two engines/card), prev checkpoints deleted each round to
# bound disk. Default base = qwen3-4b-INSTRUCT (probe-validated; the -base model emits garbage).
# FULL_PARAM=0 (default) -> the LoRA path (7B-instruct, matches the patsp_init/ adapters).
FULL_PARAM="${FULL_PARAM:-0}"
HB_D0="${HB_D0:-0.5}"                                  # training difficulty (forwarded from selfplay.sh)
if [ "$FULL_PARAM" = "1" ]; then
  export BASE_MODEL="${BASE_MODEL:-$PROJ/data/models/qwen3-4b-instruct}"
  EXP_DEFAULT="$RZERO_STORAGE/patsp_herobench_full"
  MML_DEFAULT=8192   # qwen3-4b prompts run longer on verbose HeroBench tasks (4096 overflowed at 4218)
  export RZERO_SCORE_MIN="${RZERO_SCORE_MIN:-0.0}"  # weak base -> accept all train tasks (no starve)
else
  export BASE_MODEL="${BASE_MODEL:-$PROJ/data/models/qwen2.5-7b-instruct}"
  EXP_DEFAULT="$RZERO_STORAGE/patsp_herobench"
  MML_DEFAULT=4096
fi

EXP="${EXP:-$EXP_DEFAULT}"
ROUNDS="${ROUNDS:-3}"
mkdir -p "$EXP"
D2="$RZERO_STORAGE/patsp_eval/herobench_D2.jsonl"
DRIVER="scripts/rzero_nofsdp_lora/patsp/patsp_selfplay.py"

# shared knobs
export RZERO_GEN_NUM_TASKS="${RZERO_GEN_NUM_TASKS:-9}"   # challenger tasks/round (9-task train pool)
export PATSP_MAX_STEPS="${PATSP_MAX_STEPS:-16}"
export HEROBENCH_MAX_STEPS="${HEROBENCH_MAX_STEPS:-16}"
export PATSP_MAX_REPLANS="${PATSP_MAX_REPLANS:-3}"
# HeroBench re-renders the full recipe every turn (verbose obs) -> small caps overflow on big tasks.
export RZERO_STAGE_MAX_MODEL_LEN="${RZERO_STAGE_MAX_MODEL_LEN:-$MML_DEFAULT}"
export LH_EVAL_GROUP="${LH_EVAL_GROUP:-3}"               # D2 has 4 tasks -> 12 eval episodes
export RZERO_TRAIN_LR="${RZERO_TRAIN_LR:-3e-5}"          # STABLE (1e-4 collapses)
export RZERO_TRAIN_EPOCHS="${RZERO_TRAIN_EPOCHS:-1}"
export HEROBENCH_TRAIN_EXCLUDE="${HEROBENCH_TRAIN_EXCLUDE:-0,4,9,11}"  # held-out D2 idx

echo "### HeroBench self-play | exp=$EXP rounds=$ROUNDS full_param=$FULL_PARAM model=$BASE_MODEL"
echo "### held-out D2: $(wc -l < "$D2") tasks ; train-excluded idx=$HEROBENCH_TRAIN_EXCLUDE"

# ---- R-Zero arm on GPU0 (port 8031, db artifact_8031.db) ----
CUDA_VISIBLE_DEVICES=0 HEROBENCH_PORT=8031 \
  python "$DRIVER" --name rzero --arm rzero --env herobench --rounds "$ROUNDS" \
    --d0 "$HB_D0" --step 0.0 --d2_tasks "$D2" --seed 1 --exp "$EXP" --full_param "$FULL_PARAM" \
    > "$EXP/rzero.log" 2>&1 &
PID_RZ=$!
echo "### R-Zero arm  GPU0 port8031 pid=$PID_RZ -> $EXP/rzero.log"

# ---- PATSP-outcome arm on GPU1 (port 8032, db artifact_8032.db) ----
CUDA_VISIBLE_DEVICES=1 HEROBENCH_PORT=8032 \
  python "$DRIVER" --name patsp_oc --arm patsp --env herobench --rounds "$ROUNDS" \
    --d0 "$HB_D0" --step 0.0 --d2_tasks "$D2" --seed 1 --exp "$EXP" --full_param "$FULL_PARAM" \
    --credit_mode outcome --outcome_lambda 0.5 \
    > "$EXP/patsp_oc.log" 2>&1 &
PID_PA=$!
echo "### PATSP arm   GPU1 port8032 pid=$PID_PA -> $EXP/patsp_oc.log"

# set-e-tolerant: ALWAYS wait for BOTH arms (a crash in one must not orphan / SIGKILL the other
# mid-round), then propagate a combined nonzero rc so the PBS wrapper's crash-guard works.
RC_RZ=0; wait $PID_RZ || RC_RZ=$?
RC_PA=0; wait $PID_PA || RC_PA=$?
echo "### DONE rzero_rc=$RC_RZ patsp_rc=$RC_PA"
echo "### rzero rounds:";  cat "$EXP/rzero/rounds.json"    2>/dev/null || echo "(none)"
echo "### patsp rounds:";  cat "$EXP/patsp_oc/rounds.json" 2>/dev/null || echo "(none)"
RC=0; [ "$RC_RZ" -ne 0 ] && RC=$RC_RZ; [ "$RC_PA" -ne 0 ] && RC=$RC_PA
exit $RC
