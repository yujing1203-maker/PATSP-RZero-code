"""Blackwell sm_120 env verification for the R-Zero/verl 'agent' env."""
import sys, traceback
ok, fail = [], []
def check(name, fn):
    try:
        msg = fn()
        ok.append((name, msg)); print(f"[PASS] {name}: {msg}")
    except Exception as e:
        fail.append((name, repr(e)))
        print(f"[FAIL] {name}: {e!r}")
        traceback.print_exc(limit=2)

# 1. torch + sm_120 + real GPU compute
def _torch():
    import torch
    al = torch.cuda.get_arch_list()
    assert 'sm_120' in al, f"sm_120 missing: {al}"
    assert torch.cuda.is_available(), "cuda not available"
    cap = torch.cuda.get_device_capability(0)
    assert cap == (12, 0), f"unexpected cap {cap}"
    return f"{torch.__version__} cuda={torch.version.cuda} cap={cap} name={torch.cuda.get_device_name(0)}"
check("torch+sm120", _torch)

def _gpu_compute():
    import torch
    x = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
    y = (x @ x).float().sum()
    torch.cuda.synchronize()
    # also exercise a non-trivial kernel path
    z = torch.nn.functional.scaled_dot_product_attention(
        *(torch.randn(1, 4, 64, 64, device='cuda', dtype=torch.bfloat16) for _ in range(3)))
    torch.cuda.synchronize()
    return f"bf16 matmul sum={y.item():.1f}, sdpa out={tuple(z.shape)}  <-- sm_120 kernels really execute"
check("gpu_compute(bf16 matmul + sdpa)", _gpu_compute)

# 2. vLLM + LoRARequest (the codebase's exact usage)
def _vllm():
    import vllm
    from vllm.lora.request import LoRARequest
    r = LoRARequest('solver_candidate_judge', 21, '/tmp/x')
    assert r.lora_int_id == 21
    r2 = LoRARequest('planner', 41, '/tmp/p'); r3 = LoRARequest('executor', 42, '/tmp/e')
    return f"vllm {vllm.__version__}; LoRARequest int_ids {r.lora_int_id}/{r2.lora_int_id}/{r3.lora_int_id} OK"
check("vllm+LoRARequest", _vllm)

# 3. support libs + versions
def _support():
    import transformers, peft, datasets, tensordict, ray, xformers, xgrammar, triton, accelerate
    from importlib.metadata import version as _v
    return (f"transformers {transformers.__version__}, peft {peft.__version__}, "
            f"datasets {datasets.__version__}, tensordict {tensordict.__version__}, ray {ray.__version__}, "
            f"xformers {xformers.__version__}, xgrammar {_v('xgrammar')}, triton {triton.__version__}, "
            f"accelerate {accelerate.__version__}")
check("support_libs", _support)

def _flashinfer():
    import flashinfer
    return f"flashinfer {getattr(flashinfer, '__version__', 'installed')}"
check("flashinfer", _flashinfer)

# 4. transformers / peft API surfaces the code calls
def _tf_api():
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    from transformers.modeling_utils import no_init_weights
    from transformers.utils import is_flash_attn_2_available
    return f"AutoTokenizer/AutoModelForCausalLM/no_init_weights ok; flash_attn2_avail={is_flash_attn_2_available()}"
check("transformers_api", _tf_api)

def _peft_api():
    from peft import LoraConfig, get_peft_model, PeftModel
    LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias='none',
               task_type='CAUSAL_LM', target_modules=['q_proj', 'v_proj'])
    return "LoraConfig/get_peft_model/PeftModel ok"
check("peft_api", _peft_api)

# 5. runbook GATE import line
def _gate():
    import torch, transformers, peft, vllm, datasets, mathruler, pyarrow, numpy
    return "runbook GATE imports all present"
check("runbook_gate_imports", _gate)

# 6. repo-specific hard deps (found by scanning the code)
def _repo_specific():
    from mathruler.grader import extract_boxed_content, grade_answer
    import math_verify, stopit, flask, omegaconf, codetiming, einops, sklearn, nltk, matplotlib, openai, wandb, torchdata
    # exercise the math grader the verifier relies on
    boxed = extract_boxed_content(r"The answer is \boxed{42}.")
    return f"mathruler.extract_boxed_content -> {boxed!r}; math_verify/stopit/flask/omegaconf/codetiming/einops/sklearn/nltk/matplotlib/openai/wandb/torchdata all import"
check("repo_specific_deps", _repo_specific)

# 7. the in-tree verl package imports (DataProto / protocol uses tensordict + ray)
def _verl():
    import verl
    from verl import protocol
    from verl.protocol import DataProto
    return f"verl imports ok (DataProto available)"
check("verl_intree_package", _verl)

print("\n" + "=" * 60)
print(f"RESULT: {len(ok)} passed, {len(fail)} failed")
if fail:
    print("FAILURES:")
    for n, e in fail:
        print(f"  - {n}: {e}")
    sys.exit(1)
print("ALL VERIFICATION CHECKS PASSED")
