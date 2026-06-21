# Dataset downloaders (`scripts/datasets/`)

Idempotent downloaders that prime the HuggingFace cache with **every** dataset
the R-Zero (no-FSDP LoRA) codebase references. Running these ahead of time means
the runtime `load_dataset(...)` calls in evaluation, training, and the Challenger
persona path hit the local cache instead of the network.

The exact ids / configs / splits were verified against:
- `evaluation/datasets_loader.py` (the math + reasoning eval suite)
- `verl/utils/dataset.py` (PersonaHub, for the Challenger persona)
- `scripts/rzero_nofsdp_lora/build_external_math_eval_parquets.py`
- `scripts/data/build_multistep_train.py` (GSM8K **train**, with `<<a op b=c>>` steps)

## Files

| File | What it does |
| --- | --- |
| `download_all.sh` | Master idempotent downloader. Builds a selection set, calls `download_one.py` per dataset, prints an OK/SKIP/FAIL manifest. |
| `download_one.py` | Per-dataset worker: `datasets.load_dataset(id, config, split)` (with retries) plus the MATH-500 CSV direct download. CLI `--name <key>`. |
| `datasets_manifest.json` | Machine-readable list of every dataset: key, hf_id, config, split, gated?, role, and the loader that consumes it. |

## How to run

```bash
# Full eval suite incl. gated GPQA (gpqa SKIPs cleanly if you have no HF auth):
bash scripts/datasets/download_all.sh

# Only the math/reasoning eval suite (no persona, no gated gpqa):
bash scripts/datasets/download_all.sh --eval-only

# Everything, including the PersonaHub persona dataset:
bash scripts/datasets/download_all.sh --full

# Add one dataset to the current set:
bash scripts/datasets/download_all.sh --eval-only --include personahub

# Opt out of a single dataset (--no-<key>):
bash scripts/datasets/download_all.sh --full --no-bbeh --no-super_gpqa

# Inspect without downloading:
bash scripts/datasets/download_all.sh --list      # list keys
bash scripts/datasets/download_all.sh --dry-run   # print the per-dataset plan

# Fetch a single dataset directly:
python3 scripts/datasets/download_one.py --name gsm8k
python3 scripts/datasets/download_one.py --list
```

### Selection model
- `--eval-only` -> the math/reasoning eval suite only.
- *(default, no mode flag)* -> the full eval suite **including** gated `gpqa`.
- `--full` -> the default set **plus** `personahub`.
- `--include <name>` adds a key; `--no-<name>` removes one. Applied after the mode.
- `RZERO_DOWNLOAD_FULL=1` is equivalent to passing `--full`.

## Dataset roles

| Role | Datasets |
| --- | --- |
| **TRAIN** | `gsm8k` (train split; `build_multistep_train.py` turns the `<<a op b=c>>` calculator annotations into ordered checkpoints) |
| **EVAL** | `gsm8k` (test), `math500`, `amc23`, `minerva`, `olympiad`, `aime2024`, `aime2025`, `mmlu_pro`, `bbeh`, `super_gpqa`, `gpqa` |
| **PERSONA** | `personahub` (`proj-persona/PersonaHub`, config `math`) — sampled by the Challenger persona prompt in `verl/utils/dataset.py` |

`gsm8k` is the only dataset used for **both** training and eval. Everything else
is eval, except `personahub` which is persona-only.

## HuggingFace auth for gated GPQA

`gpqa` (`Idavidrein/gpqa`, config `gpqa_diamond`) is **GATED**. Without auth,
`download_all.sh` marks it **SKIP** (it does **not** abort the run). To fetch it:

```bash
huggingface-cli login          # or: export HF_TOKEN=hf_xxx
# then accept the license at https://huggingface.co/datasets/Idavidrein/gpqa
bash scripts/datasets/download_all.sh --include gpqa
```

`download_one.py` detects auth via `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` /
`HUGGINGFACE_HUB_TOKEN`, the `huggingface_hub` saved token, or `$HF_HOME/token`,
and exits with code `3` (-> SKIP) when none is present.

## Offline / cache usage

Both scripts honor the standard HF cache env vars:

- `HF_HOME` — cache root. Defaults to `~/.cache/huggingface`. `download_all.sh`
  exports it and `mkdir -p`s it.
- `HF_DATASETS_CACHE` — optional explicit datasets cache dir (printed if set).

Pre-download once with the cache root you intend to use, then run the pipeline
fully offline against the same `HF_HOME`:

```bash
HF_HOME=/data/hf_cache bash scripts/datasets/download_all.sh --full
# later, offline:
HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 bash run_pipeline.sh
```

Re-running `download_all.sh` is safe: `load_dataset` reuses the cache, so a
second run only re-validates. The MATH-500 primary source is the openaipublic
CSV (read live at runtime); `download_one.py` verifies its reachability and also
warms the `HuggingFaceH4/MATH-500` HF mirror.

## Relationship to `scripts/data/`

`scripts/data/download_datasets.sh` is now a thin wrapper that forwards to
`scripts/datasets/download_all.sh`. The build scripts in `scripts/data/`
(`build_multistep_train.*`, `build_eval_parquets.sh`) are **processing** steps
(they turn downloaded datasets into parquet); they are unchanged.
