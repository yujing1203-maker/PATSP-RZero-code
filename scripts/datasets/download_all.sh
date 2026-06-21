#!/usr/bin/env bash
# Master, idempotent downloader for EVERY dataset the R-Zero (no-FSDP LoRA)
# codebase references. Primes the HuggingFace cache (honoring $HF_HOME /
# $HF_DATASETS_CACHE) by calling scripts/datasets/download_one.py per dataset.
#
# Datasets (verified against evaluation/datasets_loader.py, verl/utils/dataset.py,
# scripts/rzero_nofsdp_lora/build_external_math_eval_parquets.py):
#   TRAIN+EVAL : gsm8k           openai/gsm8k (main)
#   EVAL       : math500         openaipublic CSV + HuggingFaceH4/MATH-500 (test)
#                                fallback lighteval/MATH (all, test)
#   EVAL       : amc23           zwhe99/amc23 (test)
#   EVAL       : minerva         zwhe99/simplerl-minerva-math (test)
#   EVAL       : olympiad        zwhe99/simplerl-OlympiadBench (test)
#   EVAL       : aime2024        HuggingFaceH4/aime_2024 (train)
#   EVAL       : aime2025        yentinglin/aime_2025 (default)
#   EVAL       : mmlu_pro        TIGER-Lab/MMLU-Pro (test)
#   EVAL       : bbeh            MrLight/bbeh-eval (train)
#   EVAL       : super_gpqa      m-a-p/SuperGPQA (train)
#   EVAL       : gpqa            Idavidrein/gpqa (gpqa_diamond, train)  [GATED]
#   PERSONA    : personahub      proj-persona/PersonaHub (math, train)
#
# Idempotent: load_dataset reuses the HF cache, so re-running only re-validates.
#
# Usage:
#   bash scripts/datasets/download_all.sh                 # default = full eval suite
#   bash scripts/datasets/download_all.sh --eval-only     # only the math eval suite
#   bash scripts/datasets/download_all.sh --full          # everything incl. personahub
#   bash scripts/datasets/download_all.sh --include gpqa  # add one dataset to the set
#   bash scripts/datasets/download_all.sh --no-bbeh       # opt out of one dataset
#   bash scripts/datasets/download_all.sh --list          # list keys, do nothing
#   bash scripts/datasets/download_all.sh --dry-run       # print the plan, fetch nothing
#   bash scripts/datasets/download_all.sh --help
#
# Selection model:
#   * A "selection set" is built from the mode flag, then mutated by
#     --include <name> (add) and --no-<name> (remove).
#   * --eval-only  : the math/reasoning eval suite (no persona, no gated gpqa).
#   * (default)    : the full eval suite incl. gated gpqa (gpqa SKIPs w/o auth).
#   * --full       : default set PLUS personahub.
#
# Env:
#   HF_HOME               HF cache root. Defaults to ~/.cache/huggingface.
#   HF_DATASETS_CACHE     Optional explicit datasets cache dir.
#   RZERO_DOWNLOAD_FULL   If "1"/"true", behaves as if --full was passed.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
DOWNLOAD_ONE="${SCRIPT_DIR}/download_one.py"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# --------------------------------------------------------------------------- #
# Canonical key sets. Keep in sync with download_one.py's DATASETS registry and
# datasets_manifest.json.
# --------------------------------------------------------------------------- #
# Eval-suite math/reasoning datasets (no persona, no gated gpqa).
EVAL_KEYS=(
    "gsm8k"
    "math500"
    "amc23"
    "minerva"
    "olympiad"
    "aime2024"
    "aime2025"
    "mmlu_pro"
    "bbeh"
    "super_gpqa"
)
# Gated datasets that are part of the eval suite but may SKIP without HF auth.
GATED_KEYS=("gpqa")
# Persona dataset (Challenger persona prompt).
PERSONA_KEYS=("personahub")

print_help () {
    sed -n '2,62p' "${BASH_SOURCE[0]}"
}

print_list () {
    "${PYTHON_BIN}" "${DOWNLOAD_ONE}" --list
}

# --------------------------------------------------------------------------- #
# Argument parsing.
# --------------------------------------------------------------------------- #
MODE="default"          # default | eval-only | full
DRY_RUN=0
declare -a INCLUDES=()  # --include <name>
declare -a EXCLUDES=()  # --no-<name>

if [ "${RZERO_DOWNLOAD_FULL:-0}" = "1" ] || [ "${RZERO_DOWNLOAD_FULL:-}" = "true" ]; then
    MODE="full"
fi

while [ "$#" -gt 0 ]; do
    arg="$1"
    case "$arg" in
        -h|--help)
            print_help
            exit 0
            ;;
        --list)
            print_list
            exit 0
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        --eval-only)
            MODE="eval-only"
            ;;
        --full)
            MODE="full"
            ;;
        --include)
            shift
            if [ "$#" -eq 0 ]; then
                echo "[download_all] ERROR: --include requires a dataset name." >&2
                exit 2
            fi
            INCLUDES+=("$1")
            ;;
        --include=*)
            INCLUDES+=("${arg#--include=}")
            ;;
        --no-*)
            EXCLUDES+=("${arg#--no-}")
            ;;
        *)
            echo "[download_all] unknown argument: $arg" >&2
            echo "[download_all] try --help" >&2
            exit 2
            ;;
    esac
    shift
done

# --------------------------------------------------------------------------- #
# HF cache environment.
# --------------------------------------------------------------------------- #
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "${HF_HOME}"

echo "[download_all] HF_HOME=${HF_HOME}"
if [ -n "${HF_DATASETS_CACHE:-}" ]; then
    echo "[download_all] HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
fi
echo "[download_all] mode=${MODE} dry_run=${DRY_RUN}"

if [ ! -f "${DOWNLOAD_ONE}" ]; then
    echo "[download_all] ERROR: helper not found: ${DOWNLOAD_ONE}" >&2
    exit 1
fi

# --------------------------------------------------------------------------- #
# Build the selection set from the mode, then apply include/exclude.
# --------------------------------------------------------------------------- #
declare -a SELECTION=()

case "$MODE" in
    eval-only)
        SELECTION=("${EVAL_KEYS[@]}")
        ;;
    default)
        SELECTION=("${EVAL_KEYS[@]}" "${GATED_KEYS[@]}")
        ;;
    full)
        SELECTION=("${EVAL_KEYS[@]}" "${GATED_KEYS[@]}" "${PERSONA_KEYS[@]}")
        ;;
    *)
        echo "[download_all] ERROR: unknown mode ${MODE}" >&2
        exit 2
        ;;
esac

# Helper: is value $1 present in the remaining args?
contains () {
    local needle="$1"; shift
    local x
    for x in "$@"; do
        [ "$x" = "$needle" ] && return 0
    done
    return 1
}

# Apply --include (dedupe).
if [ "${#INCLUDES[@]}" -gt 0 ]; then
    for inc in "${INCLUDES[@]}"; do
        if ! contains "$inc" "${SELECTION[@]}"; then
            SELECTION+=("$inc")
        fi
    done
fi

# Apply --no-<name> (rebuild list without excluded keys).
if [ "${#EXCLUDES[@]}" -gt 0 ]; then
    declare -a FILTERED=()
    for key in "${SELECTION[@]}"; do
        if contains "$key" "${EXCLUDES[@]}"; then
            echo "[download_all] opt-out: skipping ${key}"
            continue
        fi
        FILTERED+=("$key")
    done
    SELECTION=("${FILTERED[@]}")
fi

if [ "${#SELECTION[@]}" -eq 0 ]; then
    echo "[download_all] ERROR: empty selection after include/exclude." >&2
    exit 2
fi

echo "[download_all] selection (${#SELECTION[@]}): ${SELECTION[*]}"

# --------------------------------------------------------------------------- #
# Fetch loop. Per-dataset OK / SKIP / FAIL classification (exit 3 -> SKIP gated).
# --------------------------------------------------------------------------- #
declare -a OK_LIST=()
declare -a SKIP_LIST=()
declare -a FAIL_LIST=()

for key in "${SELECTION[@]}"; do
    echo "------------------------------------------------------------"
    echo "[download_all] dataset=${key}"
    CMD=("${PYTHON_BIN}" "${DOWNLOAD_ONE}" --name "${key}")
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[download_all] DRY-RUN cmd: ${CMD[*]}"
        OK_LIST+=("${key} (dry-run)")
        continue
    fi

    # Run without aborting the whole script on a single failure.
    set +e
    "${CMD[@]}"
    rc=$?
    set -e

    case "$rc" in
        0)
            OK_LIST+=("${key}")
            ;;
        3)
            # download_one signals "gated + no auth" -> SKIP, not fatal.
            SKIP_LIST+=("${key} (gated; HF auth required)")
            ;;
        *)
            FAIL_LIST+=("${key} (exit ${rc})")
            ;;
    esac
done

# --------------------------------------------------------------------------- #
# Manifest.
# --------------------------------------------------------------------------- #
echo "============================================================"
echo "[download_all] MANIFEST (HF_HOME=${HF_HOME})"
echo "[download_all] OK   (${#OK_LIST[@]}):"
if [ "${#OK_LIST[@]}" -gt 0 ]; then
    for x in "${OK_LIST[@]}"; do echo "  + ${x}"; done
fi
echo "[download_all] SKIP (${#SKIP_LIST[@]}):"
if [ "${#SKIP_LIST[@]}" -gt 0 ]; then
    for x in "${SKIP_LIST[@]}"; do echo "  ~ ${x}"; done
fi
echo "[download_all] FAIL (${#FAIL_LIST[@]}):"
if [ "${#FAIL_LIST[@]}" -gt 0 ]; then
    for x in "${FAIL_LIST[@]}"; do echo "  - ${x}"; done
fi

if [ "${#SKIP_LIST[@]}" -gt 0 ]; then
    echo "[download_all] NOTE: skipped datasets are GATED. To fetch them:"
    echo "[download_all]   1) huggingface-cli login   (or export HF_TOKEN=...)"
    echo "[download_all]   2) accept the dataset license on its HuggingFace page"
    echo "[download_all]   3) re-run with --include <name>"
fi

# A FAIL (genuine error) is fatal; a SKIP (gated, no auth) is not.
if [ "${#FAIL_LIST[@]}" -gt 0 ]; then
    echo "[download_all] done with FAILURES." >&2
    exit 1
fi

echo "[download_all] done."
