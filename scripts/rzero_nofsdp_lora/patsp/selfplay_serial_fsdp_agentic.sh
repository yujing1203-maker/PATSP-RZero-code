#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Serial full-parameter FSDP self-play driver for:
#   ROUTE=rzero : challenger + monolithic solver
#   ROUTE=patsp : challenger + planner + executor
#
# Typical usage:
#   ROUTE=rzero ROUNDS=2 CUDA_DEVICES=0,1 bash scripts/rzero_nofsdp_lora/patsp/selfplay_serial_fsdp_agentic.sh
#   ROUTE=patsp ROUNDS=1 CUDA_DEVICES=0,1 bash scripts/rzero_nofsdp_lora/patsp/selfplay_serial_fsdp_agentic.sh
# ============================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-agent}"

cd "${PROJECT_DIR:-/home/yj/projects/PATSP-RZero-code-20260611/PATSP-RZero-code-20260611}"

export RZERO_PROJECT="$PWD"
export RZERO_STORAGE="${RZERO_STORAGE:-$HOME/RZero_storage}"
export RZERO_RUNS="${RZERO_RUNS:-$RZERO_STORAGE/runs}"
export TMPDIR="${TMPDIR:-$RZERO_STORAGE/tmp}"
export BASE_MODEL="${BASE_MODEL:-$RZERO_PROJECT/data/models/qwen3-4b-instruct}"
export PYTHONPATH="$RZERO_PROJECT/scripts/rzero_nofsdp_lora:$RZERO_PROJECT:${PYTHONPATH:-}"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

mkdir -p "$RZERO_RUNS" "$TMPDIR"

ROUTE="${ROUTE:-rzero}"
ENV_NAME="${ENV_NAME:-textcraft}"
ROUNDS="${ROUNDS:-1}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
FSDP_NPROC="${FSDP_NPROC:-2}"

SEED="${SEED:-3001}"
OUT_DIR="${OUT_DIR:-$RZERO_RUNS/${ROUTE}_${ENV_NAME}_full_param_fsdp_rounds${ROUNDS}_seed${SEED}}"

# Challenger/task generation knobs.
CHALLENGER_NUM_PROMPTS="${CHALLENGER_NUM_PROMPTS:-1}"
CHALLENGER_SAMPLES_PER_PROMPT="${CHALLENGER_SAMPLES_PER_PROMPT:-4}"
SOLVER_TASK_SAMPLES_PER_PROMPT="${SOLVER_TASK_SAMPLES_PER_PROMPT:-4}"
DIFFICULTY="${DIFFICULTY:-0.5}"
PROBE_K="${PROBE_K:-1}"
SCORE_MIN="${SCORE_MIN:-0.0}"
SCORE_MAX="${SCORE_MAX:-1.0}"

# Rollout knobs.
GROUP_SIZE="${GROUP_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-20}"
MAX_REPLANS="${MAX_REPLANS:-3}"

# Training knobs.
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
TRAIN_MAX_ROWS="${TRAIN_MAX_ROWS:-64}"
TRAIN_MAX_SEQ_LEN="${TRAIN_MAX_SEQ_LEN:-4096}"
TRAIN_LR="${TRAIN_LR:-2e-6}"

# vLLM knobs.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"

if [[ "$ROUTE" != "rzero" && "$ROUTE" != "patsp" ]]; then
  echo "ERROR: ROUTE must be rzero or patsp, got: $ROUTE" >&2
  exit 2
fi

echo "============================================================"
echo "Serial full-param FSDP self-play"
echo "route                  = $ROUTE"
echo "env                    = $ENV_NAME"
echo "rounds                 = $ROUNDS"
echo "cuda devices           = $CUDA_DEVICES"
echo "fsdp nproc             = $FSDP_NPROC"
echo "base model             = $BASE_MODEL"
echo "out dir                = $OUT_DIR"
echo "seed                   = $SEED"
echo "group size             = $GROUP_SIZE"
echo "max steps              = $MAX_STEPS"
echo "gpu_memory_utilization = $GPU_MEMORY_UTILIZATION"
echo "============================================================"

echo
echo "== git state =="
git branch --show-current || true
git log --oneline -5 || true
git status --short || true

echo
echo "== cleanup old vLLM =="
pkill -f "VLLM::EngineCore" || true
sleep 5
nvidia-smi || true

echo
echo "== remove old OUT_DIR =="
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

COMMON_ARGS=(
  --route "$ROUTE"
  --full_param
  --train_backend fsdp
  --fsdp_nproc "$FSDP_NPROC"
  --env "$ENV_NAME"
  --policy vllm
  --base_model "$BASE_MODEL"
  --rounds "$ROUNDS"
  --challenger_num_prompts "$CHALLENGER_NUM_PROMPTS"
  --challenger_samples_per_prompt "$CHALLENGER_SAMPLES_PER_PROMPT"
  --solver_task_samples_per_prompt "$SOLVER_TASK_SAMPLES_PER_PROMPT"
  --difficulty "$DIFFICULTY"
  --probe_k "$PROBE_K"
  --score_min "$SCORE_MIN"
  --score_max "$SCORE_MAX"
  --group_size "$GROUP_SIZE"
  --max_steps "$MAX_STEPS"
  --max_replans "$MAX_REPLANS"
  --train_epochs "$TRAIN_EPOCHS"
  --train_max_rows "$TRAIN_MAX_ROWS"
  --train_max_seq_len "$TRAIN_MAX_SEQ_LEN"
  --train_lr "$TRAIN_LR"
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
  --max_tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --out_dir "$OUT_DIR"
  --seed "$SEED"
)

echo
echo "== run protocol =="
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
python scripts/rzero_nofsdp_lora/patsp/selfplay_protocol_v1.py "${COMMON_ARGS[@]}"

echo
echo "== rounds.json =="
cat "$OUT_DIR/rounds.json" | python -m json.tool

echo
echo "== role/route contract check =="
python - <<PY
import json
from pathlib import Path
from collections import Counter

out = Path("$OUT_DIR")
route = "$ROUTE"
rounds = json.loads((out / "rounds.json").read_text())
assert rounds["route"] == route, rounds["route"]
assert rounds["rounds"], "no rounds"

for rec in rounds["rounds"]:
    print("\\n== round", rec["round"], "==")
    assert rec["route"] == route, rec.get("route")

    if route == "rzero":
        assert "solver_out" in rec, rec.keys()
        assert "solver_rows_jsonl" in rec, rec.keys()
        assert "planner_out" not in rec
        assert "executor_out" not in rec

        crows = [json.loads(x) for x in Path(rec["challenger_rows_jsonl"]).read_text().splitlines() if x.strip()]
        srows = [json.loads(x) for x in Path(rec["solver_rows_jsonl"]).read_text().splitlines() if x.strip()]
        assert crows, "empty challenger rows"
        assert srows, "empty solver rows"

        assert all(r.get("role") == "challenger" for r in crows)
        assert all(r.get("role") == "solver" for r in srows)

        print("challenger_rows =", len(crows))
        print("solver_rows     =", len(srows))
        print("challenger_out  =", rec.get("challenger_out"))
        print("solver_out      =", rec.get("solver_out"))
        print("trained flags   =", {
            "challenger": rec.get("challenger_trained"),
            "solver": rec.get("solver_trained"),
        })

    elif route == "patsp":
        for bad in ["solver_out", "solver_rows_jsonl", "solver_traj_jsonl"]:
            assert bad not in rec, bad

        for key in [
            "challenger_out",
            "planner_out",
            "executor_out",
            "challenger_rows_jsonl",
            "planner_rows_jsonl",
            "executor_rows_jsonl",
            "patsp_traj_jsonl",
        ]:
            assert key in rec, key

        checks = [
            ("challenger_rows_jsonl", "challenger"),
            ("planner_rows_jsonl", "planner"),
            ("executor_rows_jsonl", "executor"),
        ]
        for key, role in checks:
            rows = [json.loads(x) for x in Path(rec[key]).read_text().splitlines() if x.strip()]
            assert rows, f"empty {key}"
            assert all(r.get("role") == role for r in rows), key
            print(key, "rows =", len(rows))

        traj = [json.loads(x) for x in Path(rec["patsp_traj_jsonl"]).read_text().splitlines() if x.strip()]
        role_counts = Counter()
        for tr in traj:
            for turn in tr.get("turns", []):
                role = turn.get("role")
                role_counts[role] += 1
                assert role in {"planner", "executor"}, role
        print("trajectory role_counts =", dict(role_counts))
        print("trained flags =", {
            "challenger": rec.get("challenger_trained"),
            "planner": rec.get("planner_trained"),
            "executor": rec.get("executor_trained"),
        })

print("\\nOK: route contract passed")
PY

echo
echo "== checkpoint meta summary =="
python - <<PY
import json
from pathlib import Path

out = Path("$OUT_DIR")
route = "$ROUTE"
rounds = json.loads((out / "rounds.json").read_text())

for rec in rounds["rounds"]:
    print("\\n== round", rec["round"], "==")
    roles = ["challenger", "solver"] if route == "rzero" else ["challenger", "planner", "executor"]
    for role in roles:
        key = f"{role}_out"
        p = Path(rec[key])
        print("\\n--", role, "--")
        print("out =", p)
        meta_path = p / "rzero_fsdp_grpo_meta.json"
        if not meta_path.exists():
            print("no meta: probably carried forward")
            continue
        meta = json.loads(meta_path.read_text())
        for k in ["raw_rows", "selected_rows", "usable_examples", "update_steps", "trainable_abs_sum_delta", "world_size"]:
            print(k, "=", meta.get(k))
PY

echo
echo "OK: serial full-param FSDP self-play finished"
