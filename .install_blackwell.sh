#!/usr/bin/env bash
# Blackwell sm_120 install for the R-Zero/verl "agent" (py3.11) conda env.
# Prebuilt wheels only; hard sm_120 gates between steps. Auto-generated.
set -uo pipefail
PIP="${CONDA_PREFIX:?activate your conda env first: conda activate agent}/bin/pip"
PY="${CONDA_PREFIX:?activate your conda env first: conda activate agent}/bin/python"

step() { echo; echo "===================== $* ====================="; }
gate() {
  $PY - "$1" <<'PYEOF'
import sys, torch
al = torch.cuda.get_arch_list()
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("arch_list", al)
assert "sm_120" in al, "FATAL: sm_120 missing from torch arch list (%s) -- %s" % (al, sys.argv[1])
print("cxx11abi", torch._C._GLIBCXX_USE_CXX11_ABI)
print("GATE OK (%s): sm_120 present" % sys.argv[1])
PYEOF
}

step "STEP 1/3  torch trio (cu128, sm_120)"
$PIP install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128 || { echo "STEP1 FAILED"; exit 11; }
gate "after-torch" || { echo "sm_120 GATE FAILED after torch"; exit 21; }

step "STEP 2/3  vllm[flashinfer]==0.11.0"
$PIP install "vllm[flashinfer]==0.11.0" \
  --extra-index-url https://download.pytorch.org/whl/cu128 || { echo "STEP2 FAILED"; exit 12; }
gate "after-vllm" || { echo "sm_120 GATE FAILED after vllm (torch got downgraded)"; exit 22; }

step "STEP 3/3  pin codebase-verified support libs"
$PIP install transformers==4.57.6 peft==0.19.1 datasets==4.8.5 \
  tensordict==0.12.2 accelerate==1.13.0 || { echo "STEP3 FAILED"; exit 13; }
gate "after-support" || { echo "sm_120 GATE FAILED after support libs"; exit 23; }

step "CORE INSTALL COMPLETE"
echo "ALL_CORE_STEPS_OK"
