#!/usr/bin/env bash
# Thin wrapper -> scripts/datasets/download_all.sh (the consolidated downloader).
#
# Dataset download logic now lives in scripts/datasets/. This wrapper is kept so
# existing call sites / docs that reference scripts/data/download_datasets.sh
# keep working.
#
# Backward-compatible argument mapping:
#   (no args)   core eval set: gsm8k + math500 only (the historical default).
#               Implemented as --eval-only with the extended benchmarks opted out.
#   --full      the full eval suite              -> download_all.sh --full
#   --help/-h   show this help
#
# Any other flags are forwarded verbatim to download_all.sh, so you can also do:
#   bash scripts/data/download_datasets.sh --eval-only
#   bash scripts/data/download_datasets.sh --include gpqa
#
# Env:
#   HF_HOME, HF_DATASETS_CACHE, RZERO_DOWNLOAD_FULL  (honored by download_all.sh)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
MASTER="${SCRIPT_DIR}/../datasets/download_all.sh"

if [ ! -f "${MASTER}" ]; then
    echo "[download_datasets] ERROR: master downloader not found: ${MASTER}" >&2
    exit 1
fi

echo "[download_datasets] forwarding to ${MASTER}"

# No arguments: preserve the historical "core only" default (gsm8k + math500)
# by starting from --eval-only and opting out of the extended benchmarks.
if [ "$#" -eq 0 ]; then
    exec bash "${MASTER}" --eval-only \
        --no-amc23 --no-minerva --no-olympiad \
        --no-aime2024 --no-aime2025 \
        --no-mmlu_pro --no-bbeh --no-super_gpqa
fi

case "${1:-}" in
    -h|--help)
        sed -n '2,30p' "${BASH_SOURCE[0]}"
        echo "----------------------------------------"
        echo "Underlying master help:"
        exec bash "${MASTER}" --help
        ;;
    *)
        # Forward all flags verbatim (e.g. --full, --eval-only, --include x).
        exec bash "${MASTER}" "$@"
        ;;
esac
