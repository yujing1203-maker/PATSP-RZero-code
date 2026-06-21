#!/usr/bin/env bash
# One-click progress view: per-experiment round scores + GPU. Instant, read-only.
# Usage: bash scripts/run/status.sh [EXP_DIR ...]     (default: every patsp_* dir in $RZERO_STORAGE)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
DIRS=("$@")
if [ ${#DIRS[@]} -eq 0 ]; then
  shopt -s nullglob
  DIRS=("$RZERO_STORAGE"/patsp_*/)
  shopt -u nullglob
fi

echo "== experiments =="
[ ${#DIRS[@]} -eq 0 ] && echo "  (no patsp_* experiments in $RZERO_STORAGE)"
for d in "${DIRS[@]}"; do
  d="${d%/}"
  case "${d##*/}" in patsp_eval|patsp_init) continue ;; esac   # not experiments
  has=0
  for a in rzero patsp_oc patsp; do
    [ -f "$d/$a/rounds.json" ] || continue
    has=1
    python - "$d" "$a" <<'PY'
import json, sys
d, a = sys.argv[1], sys.argv[2]
r = json.load(open(f"{d}/{a}/rounds.json"))
xs = [(x["round"], x["D2"]) for x in r]
mean = round(sum(x["D2"] for x in r)/len(r), 4) if r else 0
print(f"  {d.split('/')[-1]:34s} {a:9s} {len(r)} rounds  mean={mean}  {xs}")
PY
  done
  [ $has -eq 0 ] && echo "  ${d##*/}: (no rounds.json yet)"
done
echo
echo "== GPU (this node) =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  /' \
  || echo "  (nvidia-smi unavailable)"
