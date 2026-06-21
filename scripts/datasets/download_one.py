#!/usr/bin/env python3
"""
Pre-fetch ONE dataset into the local HuggingFace cache (or download the MATH-500
CSV directly), with retries and clear messaging.

This is the per-dataset worker invoked by ``download_all.sh``. It deliberately
mirrors the exact ids / configs / splits that the codebase loads at runtime so
that priming the cache here means runtime ``load_dataset(...)`` calls hit the
cache instead of the network. The authoritative consumers are:

  * evaluation/datasets_loader.py          (the math + reasoning eval suite)
  * verl/utils/dataset.py                  (PersonaHub, for the Challenger persona)
  * scripts/rzero_nofsdp_lora/build_external_math_eval_parquets.py
  * scripts/data/build_multistep_train.py  (GSM8K train, with <<a op b=c>> steps)

Usage:
    python3 download_one.py --name gsm8k
    python3 download_one.py --name math500
    python3 download_one.py --list
    python3 download_one.py --name gpqa            # GATED: needs HF auth

Exit codes:
    0   success
    3   gated dataset and no usable HF auth detected -> caller should SKIP
    1   any other failure (after retries)
"""

import argparse
import os
import sys
import time
import urllib.request

# --------------------------------------------------------------------------- #
# Dataset registry. One entry per key. This is the single source of truth that
# download_all.sh and datasets_manifest.json are kept consistent with.
#
# Fields:
#   hf_id   : HuggingFace dataset id, or None for a direct-URL (CSV) dataset.
#   config  : load_dataset config / name (a.k.a. "subset"); None if not used.
#   split   : split to materialize; None means "all splits".
#   gated   : True if the dataset requires accepting a license / HF auth.
#   url     : direct download URL (only for the MATH-500 CSV primary source).
#   role    : "train" | "eval" | "persona" (what the dataset is used for).
#   note    : free-form description.
# --------------------------------------------------------------------------- #
DATASETS = {
    "gsm8k": {
        "hf_id": "openai/gsm8k",
        "config": "main",
        "split": None,  # both train (multistep source) and test (eval) are used
        "gated": False,
        "url": None,
        "role": "train+eval",
        "note": "GSM8K; train answers carry <<a op b=c>> calc annotations.",
    },
    "math500": {
        # PRIMARY source is the openaipublic CSV (datasets_loader.MathDatasetHandler).
        # The HF mirror + fallback are tried by download_all.sh / the build scripts.
        "hf_id": "HuggingFaceH4/MATH-500",
        "config": None,
        "split": "test",
        "gated": False,
        "url": "https://openaipublic.blob.core.windows.net/simple-evals/math_500_test.csv",
        "role": "eval",
        "note": "MATH-500; primary = openaipublic CSV, HF mirror also cached.",
    },
    "math500_fallback": {
        "hf_id": "lighteval/MATH",
        "config": "all",
        "split": "test",
        "gated": False,
        "url": None,
        "role": "eval",
        "note": "Fallback for MATH-500 when the HF mirror is unavailable.",
    },
    "amc23": {
        "hf_id": "zwhe99/amc23",
        "config": None,
        "split": "test",
        "gated": False,
        "url": None,
        "role": "eval",
        "note": "AMC 2023 competition math.",
    },
    "minerva": {
        "hf_id": "zwhe99/simplerl-minerva-math",
        "config": None,
        "split": "test",
        "gated": False,
        "url": None,
        "role": "eval",
        "note": "Minerva math (SimpleRL packaging).",
    },
    "olympiad": {
        "hf_id": "zwhe99/simplerl-OlympiadBench",
        "config": None,
        "split": "test",
        "gated": False,
        "url": None,
        "role": "eval",
        "note": "OlympiadBench (SimpleRL packaging).",
    },
    "aime2024": {
        "hf_id": "HuggingFaceH4/aime_2024",
        "config": None,
        "split": "train",
        "gated": False,
        "url": None,
        "role": "eval",
        "note": "AIME 2024; the only split shipped is 'train'.",
    },
    "aime2025": {
        "hf_id": "yentinglin/aime_2025",
        "config": "default",
        "split": None,  # loader uses ['train']; cache the whole config
        "gated": False,
        "url": None,
        "role": "eval",
        "note": "AIME 2025; loader reads the 'train' split of config 'default'.",
    },
    "mmlu_pro": {
        "hf_id": "TIGER-Lab/MMLU-Pro",
        "config": None,
        "split": "test",
        "gated": False,
        "url": None,
        "role": "eval",
        "note": "MMLU-Pro multiple-choice reasoning.",
    },
    "bbeh": {
        "hf_id": "MrLight/bbeh-eval",
        "config": None,
        "split": "train",
        "gated": False,
        "url": None,
        "role": "eval",
        "note": "BIG-Bench Extra Hard eval; loader reads 'train'.",
    },
    "super_gpqa": {
        "hf_id": "m-a-p/SuperGPQA",
        "config": None,
        "split": None,  # loader iterates dataset['train']; cache all splits
        "gated": False,
        "url": None,
        "role": "eval",
        "note": "SuperGPQA; loader reads the 'train' split.",
    },
    "gpqa": {
        "hf_id": "Idavidrein/gpqa",
        "config": "gpqa_diamond",
        "split": "train",
        "gated": True,
        "url": None,
        "role": "eval",
        "note": "GPQA diamond; GATED -> requires HF login + accepted license.",
    },
    "personahub": {
        "hf_id": "proj-persona/PersonaHub",
        "config": "math",
        "split": "train",
        "gated": False,
        "url": None,
        "role": "persona",
        "note": "PersonaHub (math); used by the Challenger persona prompt.",
    },
}


def log(msg: str) -> None:
    sys.stdout.write(f"[download_one] {msg}\n")
    sys.stdout.flush()


def err(msg: str) -> None:
    sys.stderr.write(f"[download_one] {msg}\n")
    sys.stderr.flush()


def _hf_auth_present() -> bool:
    """
    Best-effort detection of usable HF auth for gated datasets.

    Checks (in order): HF token env vars, then the on-disk token saved by
    `huggingface-cli login`. We do NOT make a network call here; the caller
    treats "no auth" as a SKIP rather than a hard failure.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(var):
            return True
    try:
        from huggingface_hub import HfFolder  # type: ignore

        if HfFolder.get_token():
            return True
    except Exception:
        # huggingface_hub may be unavailable or its API may differ; fall through.
        pass
    # Default cache location for the CLI token.
    token_path = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "token",
    )
    if os.path.isfile(token_path):
        try:
            with open(token_path, encoding="utf-8") as fh:
                if fh.read().strip():
                    return True
        except Exception:
            pass
    return False


def _retry(fn, *, attempts: int = 3, base_delay: float = 3.0, what: str = "operation"):
    """Run ``fn`` with simple exponential backoff. Re-raises the last error."""
    last_exc = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we want to retry any transient error
            last_exc = exc
            if i < attempts:
                delay = base_delay * (2 ** (i - 1))
                err(f"{what}: attempt {i}/{attempts} failed: {exc!r}; retrying in {delay:.0f}s")
                time.sleep(delay)
            else:
                err(f"{what}: attempt {i}/{attempts} failed: {exc!r}; giving up")
    raise last_exc  # type: ignore[misc]


def _download_csv(url: str) -> int:
    """
    Prime the MATH-500 CSV. We fetch it once with urllib to confirm the primary
    source is reachable; runtime uses pandas.read_csv on the same URL. We do not
    persist it (the runtime reads it live), but verifying reachability here gives
    an early, clear signal.
    """

    def _do():
        req = urllib.request.Request(url, headers={"User-Agent": "rzero-download_one/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - trusted URL
            data = resp.read()
        return len(data)

    nbytes = _retry(_do, what=f"GET {url}")
    log(f"OK csv {url} bytes={nbytes}")
    return 0


def _load_hf(hf_id: str, config, split) -> int:
    from datasets import load_dataset  # imported lazily so --list works without datasets

    args = (hf_id,) if config is None else (hf_id, config)
    kwargs = {"trust_remote_code": True}
    if split is not None:
        kwargs["split"] = split

    def _do():
        return load_dataset(*args, **kwargs)

    ds = _retry(_do, what=f"load_dataset({hf_id}, config={config}, split={split})")

    # Report rows without forcing extra work.
    try:
        num_rows = getattr(ds, "num_rows", None)
        if isinstance(num_rows, dict):
            log(f"OK {hf_id} splits={dict(num_rows)}")
        elif num_rows is not None:
            log(f"OK {hf_id} rows={int(num_rows)}")
        else:
            log(f"OK {hf_id}")
    except Exception:
        log(f"OK {hf_id}")
    return 0


def download(name: str) -> int:
    if name not in DATASETS:
        err(f"unknown dataset key: {name!r}. Use --list to see valid keys.")
        return 1

    spec = DATASETS[name]
    hf_id = spec["hf_id"]
    config = spec["config"]
    split = spec["split"]
    gated = spec["gated"]
    url = spec["url"]

    log(f"name={name} hf_id={hf_id} config={config} split={split} gated={gated} role={spec['role']}")

    if gated and not _hf_auth_present():
        err(
            f"{name} ({hf_id}) is GATED and no HuggingFace auth was detected.\n"
            "[download_one]   Run `huggingface-cli login` (or set HF_TOKEN) and accept the\n"
            f"[download_one]   dataset license at https://huggingface.co/datasets/{hf_id}\n"
            "[download_one]   Skipping (exit 3)."
        )
        return 3

    # Direct-CSV datasets (MATH-500 primary). We verify the CSV AND warm the HF
    # mirror so both runtime paths are covered.
    if url is not None:
        rc_csv = _download_csv(url)
        if hf_id is not None:
            try:
                _load_hf(hf_id, config, split)
            except Exception as exc:  # noqa: BLE001
                err(f"{name}: HF mirror {hf_id} unavailable ({exc!r}); CSV primary already verified.")
        return rc_csv

    try:
        return _load_hf(hf_id, config, split)
    except Exception as exc:  # noqa: BLE001
        err(f"{name} ({hf_id}) failed after retries: {exc!r}")
        return 1


def parse_args():
    p = argparse.ArgumentParser(description="Pre-fetch one dataset for the R-Zero pipeline.")
    p.add_argument("--name", help="dataset key (see --list)")
    p.add_argument("--list", action="store_true", help="list known dataset keys and exit")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        for key, spec in DATASETS.items():
            gated = " [GATED]" if spec["gated"] else ""
            log(f"{key:18s} -> {spec['hf_id']}  config={spec['config']} split={spec['split']} role={spec['role']}{gated}")
        return 0
    if not args.name:
        err("either --name <key> or --list is required.")
        return 2
    return download(args.name)


if __name__ == "__main__":
    sys.exit(main())
