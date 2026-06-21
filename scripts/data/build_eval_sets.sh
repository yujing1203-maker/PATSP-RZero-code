#!/usr/bin/env bash
# Generate held-out D2 eval sets into $RZERO_STORAGE/patsp_eval/ (see build_eval_sets.py for details).
# Usage:
#   bash scripts/data/build_eval_sets.sh                      # all four envs
#   bash scripts/data/build_eval_sets.sh --envs textcraft,alfworld
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/../.." && pwd)"
export PYTHONPATH="$PROJ/scripts/rzero_nofsdp_lora:${PYTHONPATH:-}"
exec python3 "$HERE/build_eval_sets.py" "$@"
