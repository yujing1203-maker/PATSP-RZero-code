#!/usr/bin/env bash
# One-click model x env PROBE — the mandatory 10-minute gate before ANY long self-play run.
# Rolls 8 episodes on the env's D2 subset and prints reward distribution + sample actions + verdict.
#
# Usage:  bash scripts/run/probe.sh ENV [MODEL_DIR] [GPU]
#   ENV    textcraft | scienceworld | alfworld | herobench
#   MODEL  full checkpoint dir (default: data/models/qwen3-4b-instruct)
#   GPU    CUDA device index (default 0)
# Examples:
#   bash scripts/run/probe.sh alfworld
#   bash scripts/run/probe.sh herobench data/models/qwen3-4b-base 1
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_env_arg "${1:-}" || exit 1
ENV="$1"; MODEL="${2:-$DEFAULT_MODEL}"; GPU="${3:-0}"
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "ERROR: GPU must be a single index (got '$GPU')" >&2; exit 1; }
[ -f "$MODEL/config.json" ] || { echo "ERROR: model dir invalid: $MODEL" >&2; exit 1; }
D2="$(d2_for "$ENV")"; STEPS="$(steps_for "$ENV")"
OUT="/tmp/probe_${ENV}_$$"; mkdir -p "$OUT"
head -4 "$D2" > "$OUT/tasks.jsonl"
export HEROBENCH_PORT=$((8200 + GPU))   # per-GPU port (set unconditionally; never inherit a run's port)

echo "== probe: env=$ENV model=$(basename "$MODEL") gpu=$GPU steps=$STEPS (8 episodes) =="
CUDA_VISIBLE_DEVICES="$GPU" python "$PROJ/scripts/rzero_nofsdp_lora/build_agentic_rollouts.py" \
  --env "$ENV" --tasks_jsonl "$OUT/tasks.jsonl" --num_tasks 0 \
  --out_jsonl "$OUT/roll.jsonl" --group_size 2 --max_steps "$STEPS" --max_replans 3 \
  --policy vllm --base_model "$MODEL" \
  --temperature 0.7 --top_p 0.95 --max_tokens 256 --gpu_memory_utilization "${RZERO_PROBE_GPU_MEMORY_UTILIZATION:-0.40}" \
  > "$OUT/log" 2>&1
RC=$?
[ $RC -ne 0 ] && { echo "PROBE FAILED rc=$RC — tail of $OUT/log:"; tail -8 "$OUT/log"; exit $RC; }

# verdict python EXITS 2 on no-signal / parse failure, so `probe.sh X && selfplay.sh X` is safe.
python - "$OUT/roll.jsonl" <<'PY'
import json, sys, collections
try:
    rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    rw=[round(float(r.get("terminal_reward",0) or 0),3) for r in rows]
    nz=sum(1 for x in rw if x>0)
    print("reward distribution :", dict(collections.Counter(rw)))
    ex=[t for r in rows[:1] for t in r.get("turns",[]) if t.get("role")=="executor"][:2]
    for t in ex: print("sample executor out :", repr((t.get("completion") or "")[:90]))
except Exception as e:
    print(f"VERDICT: PARSE ERROR ({e}) — rollout output unreadable."); sys.exit(2)
print()
if nz==0:
    print(f"VERDICT: NO SIGNAL ({len(rw)} episodes, all 0) — do NOT launch self-play with this (model, env).")
    print("         garbage/repetition => model can't follow format; clean actions but 0 reward")
    print("         => difficulty too high (selfplay.sh <env> --d0 0.0) or env mismatch.")
    sys.exit(2)
print(f"VERDICT: SIGNAL OK ({nz}/{len(rw)} episodes scored > 0) — safe to launch self-play.")
PY
RC=$?
echo "(artifacts: $OUT)"
exit $RC
