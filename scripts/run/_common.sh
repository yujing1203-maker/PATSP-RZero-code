#!/usr/bin/env bash
# Shared environment setup for the one-click scripts (scripts/run/*.sh). Source me, don't run me.
# Centralizes every hard-won knob so individual scripts stay tiny. See TUTORIAL.md.

# Repo root from this file's location (scripts/run/_common.sh -> two levels up).
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RZERO_PROJECT="$PROJ"
export RZERO_STORAGE="${RZERO_STORAGE:-$HOME/RZero_storage}"
export PYTHONPATH="$PROJ/scripts/rzero_nofsdp_lora"

# conda env `agent` (the only validated Blackwell sm_120 stack: torch2.8+cu128 / vllm0.11)
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV:-}" != "agent" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null && conda activate agent 2>/dev/null || true
fi

# sm_120 mandatory flags (system gcc 8.5 cannot JIT flashinfer) + stability knobs
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RZERO_REPETITION_PENALTY="${RZERO_REPETITION_PENALTY:-1.2}"   # breaks single-token loops
export RZERO_STAGE_MAX_MODEL_LEN="${RZERO_STAGE_MAX_MODEL_LEN:-8192}" # verbose envs overflow 4096
export ALFWORLD_DATA="${ALFWORLD_DATA:-$RZERO_STORAGE/alfworld_data}"

# Default model: the 4B instruct (probe-validated on ALFWorld/HeroBench; base models degenerate).
DEFAULT_MODEL="$PROJ/data/models/qwen3-4b-instruct"

# env -> frozen D2 eval set
d2_for() {
  case "$1" in
    textcraft)    echo "$RZERO_STORAGE/patsp_eval/textcraft_D2.jsonl" ;;
    textcraft30)  echo "$RZERO_STORAGE/patsp_eval/textcraft_EVAL30.jsonl" ;;
    scienceworld) echo "$RZERO_STORAGE/patsp_eval/scienceworld_D2_easy.jsonl" ;;
    alfworld)     echo "${RZERO_ALFWORLD_D2:-$RZERO_STORAGE/patsp_eval/alfworld_D2.jsonl}" ;;
    herobench)    echo "$RZERO_STORAGE/patsp_eval/herobench_D2.jsonl" ;;
    longreason)   echo "${RZERO_LONGREASON_D2:-$RZERO_STORAGE/longreason_train_eval/longreason_8k_eval_D2_30_tasks.jsonl}" ;;
    *) return 1 ;;
  esac
}

# env -> episode step budget (probe-validated; sciworld needs navigation room)
steps_for() {
  case "$1" in
    textcraft) echo 16 ;; scienceworld) echo 24 ;; alfworld) echo 14 ;; herobench) echo 16 ;; longreason) echo 1 ;;
    *) echo 16 ;;
  esac
}

# env -> default training difficulty d0 (weak models need the easy end on sciworld)
d0_for() {
  case "$1" in
    scienceworld) echo 0.0 ;; longreason) echo 0.5 ;; *) echo 0.5 ;;
  esac
}

require_env_arg() {  # validate $1 is a known env
  case "${1:-}" in
    textcraft|scienceworld|alfworld|herobench|longreason) return 0 ;;
    *) echo "ERROR: first argument must be one of: textcraft scienceworld alfworld herobench longreason" >&2; return 1 ;;
  esac
}

mean_from_summary() {  # <dir> -> prints mean_terminal_reward from <dir>/summary.json
  python -c "import json,sys;print(round(json.load(open(sys.argv[1]+'/summary.json'))['mean_terminal_reward'],4))" "$1" 2>/dev/null
}
