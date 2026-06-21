#!/usr/bin/env bash
# Wrapper around scripts/data/build_multistep_train.py with env defaults.
#
# Builds a multi-checkpoint TRAINING parquet under
# $RZERO_EXP/train. GSM8K uses <<a op b = c>> calculator annotations as ordered
# checkpoints; MATH-500 uses a single final-boxed checkpoint.
#
# Usage:
#   bash scripts/data/build_multistep_train.sh                  # gsm8k, defaults
#   bash scripts/data/build_multistep_train.sh math500 200
#
# Positional (optional): <dataset> <n>
#
# Env:
#   RZERO_PROJECT   repo root (required)
#   RZERO_EXP       experiment storage root (required)
#   RZERO_MULTISTEP_DATASET   default dataset (gsm8k|math500), default gsm8k
#   RZERO_MULTISTEP_N         default number of problems, default 200
#   RZERO_MULTISTEP_SEED      default seed, default 3000
#   RZERO_MULTISTEP_OUT       explicit out parquet path (overrides default location)

set -euo pipefail

if [ -z "${RZERO_PROJECT:-}" ]; then
    echo "[build_multistep_train] ERROR: RZERO_PROJECT is not set." >&2
    exit 1
fi
if [ -z "${RZERO_EXP:-}" ]; then
    echo "[build_multistep_train] ERROR: RZERO_EXP is not set." >&2
    exit 1
fi

DATASET="${1:-${RZERO_MULTISTEP_DATASET:-gsm8k}}"
N="${2:-${RZERO_MULTISTEP_N:-200}}"
SEED="${RZERO_MULTISTEP_SEED:-3000}"

OUT_DIR="${RZERO_EXP}/train"
DEFAULT_OUT="${OUT_DIR}/multistep_${DATASET}_n${N}_seed${SEED}.parquet"
OUT_PARQUET="${RZERO_MULTISTEP_OUT:-${DEFAULT_OUT}}"

mkdir -p "$(dirname "${OUT_PARQUET}")"

echo "[build_multistep_train] RZERO_PROJECT=${RZERO_PROJECT}"
echo "[build_multistep_train] RZERO_EXP=${RZERO_EXP}"
echo "[build_multistep_train] dataset=${DATASET} n=${N} seed=${SEED}"
echo "[build_multistep_train] out_parquet=${OUT_PARQUET}"

cd "${RZERO_PROJECT}"

python3 "${RZERO_PROJECT}/scripts/data/build_multistep_train.py" \
    --dataset "${DATASET}" \
    --n "${N}" \
    --seed "${SEED}" \
    --out_parquet "${OUT_PARQUET}"

echo "[build_multistep_train] done -> ${OUT_PARQUET}"
