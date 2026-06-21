#!/usr/bin/env bash
# One-click environment health check. Run me FIRST on any new node/setup. No GPU work, ~10s.
# Usage: bash scripts/run/check_env.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ok=0; bad=0
pass(){ echo "  [OK]   $1"; ok=$((ok+1)); }
fail(){ echo "  [FAIL] $1"; bad=$((bad+1)); }

echo "== PATSP environment check =="
[ "$(python -c 'import sys;print(sys.prefix.endswith("agent"))' 2>/dev/null)" = "True" ] \
  && pass "conda env 'agent' active" || fail "conda env 'agent' NOT active (source ~/miniconda3/etc/profile.d/conda.sh && conda activate agent)"
python -c "import torch;assert torch.__version__.startswith('2.8')" 2>/dev/null \
  && pass "torch 2.8 (cu128)" || fail "torch 2.8+cu128 missing (see .install_blackwell.sh)"
python -c "import vllm;assert vllm.__version__.startswith('0.11')" 2>/dev/null \
  && pass "vllm 0.11" || fail "vllm 0.11 missing"
[ "${VLLM_USE_FLASHINFER_SAMPLER:-}" = "0" ] && pass "VLLM_USE_FLASHINFER_SAMPLER=0" || fail "sampler flag unset (engine will loop on flashinfer JIT)"
nvidia-smi -L >/dev/null 2>&1 && pass "GPU visible: $(nvidia-smi -L | wc -l) card(s)" || fail "no GPU visible"
[ -f "$DEFAULT_MODEL/config.json" ] && pass "default model present ($DEFAULT_MODEL)" || fail "default model missing: hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir $DEFAULT_MODEL  (set HF_HUB_DISABLE_XET=1!)"
for e in textcraft scienceworld alfworld herobench; do
  [ -f "$(d2_for $e)" ] && pass "D2 set: $e" || fail "D2 set missing for $e ($(d2_for $e))"
done
[ -d "$ALFWORLD_DATA/json_2.1.1" ] && pass "ALFWorld data" || fail "ALFWorld data missing (ALFWORLD_DATA=$ALFWORLD_DATA alfworld-download)"
[ -d "$PROJ/third_party/HeroBench/Virtual_Environment" ] && pass "HeroBench in-project" || fail "third_party/HeroBench missing"
PYTHONPATH="$PROJ/scripts/rzero_nofsdp_lora" python -c "import envs, rollout_agentic" 2>/dev/null \
  && pass "pipeline imports" || fail "pipeline import broken"
echo "== result: $ok ok, $bad failed =="
exit $([ $bad -eq 0 ] && echo 0 || echo 1)
