import json
import os
import numpy as np
from pathlib import Path
import csv

# ─── PATHS ────────────────────────────────────────────────────────────────────
try:
    REPO_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    REPO_ROOT = Path.cwd()

# If running as a script, __file__ exists; keep user's original behavior
REPO_ROOT = Path(__file__).resolve().parents[1]

PLOTS_DIR = REPO_ROOT / "results" / "plots_tables"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

difficulty = "base"
RESULTS_DIR = REPO_ROOT / "results" / f"results_{difficulty}"



# SD computation mode:
#   "per_difficulty" → SD across the 9 per-difficulty means (original behavior)
#   "all_samples"    → SD across all raw samples pooled across difficulties
sd_mode = "per_difficulty"  # change to "all_samples" to pool across all samples


# ─── MODEL DISCOVERY ──────────────────────────────────────────────────────────
def build_model_specs(results_dir: str):
    """
    Scan results_dir and build MODEL_SPECS:
      - include only files ending with .json
      - exclude *_code_logs.json and *_full_log.json
      - key = base filename without .json
      - think=True if 'think' appears in the name (case-insensitive)
    """
    specs = {}
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith(".json"):
            continue
        name = os.path.splitext(fn)[0]
        if name.endswith("_code_logs") or name.endswith("_full_log"):
            continue
        specs[name] = {
            "path": os.path.join(results_dir, fn),
            "think": ("think" in name.lower())
        }
    return specs

# ─── MODEL SPECIFICATIONS (auto-built) ────────────────────────────────────────
MODEL_SPECS = build_model_specs(str(RESULTS_DIR))

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def mean_std(arr):
    """Mean and std that ignore NaNs and safely handle empty/length-1 arrays."""
    a = np.array(arr, dtype=float)
    if a.size == 0:
        return 0.0, 0.0
    m = float(np.nanmean(a))
    s = float(np.nanstd(a))  # population SD; use ddof=1 for sample SD if desired
    if np.isnan(m): m = 0.0
    if np.isnan(s): s = 0.0
    return m, s

def analyze_results(json_path):
    """
    Returns:
      diffs                : list[float]               # difficulty values
      succs                : list[float]               # per-difficulty success rate
      scores_per_diff      : list[float]               # per-difficulty mean score
      toks_per_diff        : list[float]               # per-difficulty mean tokens
      score_samples        : list[float]               # pooled all score samples
      token_samples        : list[float]               # pooled all token samples
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    diffs, succs, scores_per_diff, toks_per_diff = [], [], [], []
    score_samples, token_samples = [], []

    for diff_str, tasks in sorted(data.items(), key=lambda x: float(x[0])):
        all_r, all_sc, all_t = [], [], []
        for td in tasks.values():
            all_r.extend(td.get("results", []))
            sc = td.get("scores", [])
            toks = td.get("tokens", [])
            all_sc.extend(sc)
            all_t.extend(toks if toks is not None else [])

            # pooled
            score_samples.extend([x for x in sc if x is not None])
            token_samples.extend([t for t in toks if t is not None])

        # success (per difficulty)
        n = len(all_r)
        wins = sum(1 for r in all_r if r == "win")
        succ = wins / n if n else 0.0

        # per-difficulty means
        avg_sc = (sum(all_sc) / len(all_sc)) if all_sc else 0.0
        valid_t = [t for t in all_t if t is not None]
        avg_t = (sum(valid_t) / len(valid_t)) if valid_t else 0.0

        diffs.append(float(diff_str))
        succs.append(succ)
        scores_per_diff.append(avg_sc)
        toks_per_diff.append(avg_t)

    return (
        diffs,
        succs,
        scores_per_diff,
        toks_per_diff,
        score_samples,
        token_samples
    )

# ─── LOAD & COMPUTE ───────────────────────────────────────────────────────────
all_metrics = {}
for name, spec in MODEL_SPECS.items():
    try:
        (diffs, succ, sc, tks, sc_samp, tk_samp) = analyze_results(spec["path"])
        all_metrics[name] = {
            "difficulty":     diffs,
            "success_rate":   succ,
            "avg_score":      sc,
            "avg_tokens":     tks,
            "score_samples":  sc_samp,   # NEW: pooled
            "token_samples":  tk_samp,   # NEW: pooled
        }
    except Exception as e:
        print(f"[WARN] Skipping {name} ({spec['path']}): {e}")

model_names = sorted(all_metrics.keys())

# ─── PRINT DETAILED PER-MODEL TABLES ─────────────────────────────────────────
for name in model_names:
    m = all_metrics[name]
    print(f"\nModel: {name}")
    print(f"{'Difficulty':>10}  {'Success':>8}  {'Score':>7}  {'Tokens':>7}")
    print("-" * 40)
    for d, s, sc, t in zip(m["difficulty"], m["success_rate"], m["avg_score"], m["avg_tokens"]):
        print(f"{d:10.2f}  {s:8.2f}  {sc:7.2f}  {t:7.2f}")
print()

# ─── SUMMARY STATISTICS (+ switchable SDs) ────────────────────────────────────
SCALING = 10000  # e.g., "per 10k tokens"
k_label = SCALING // 1000

summary_rows = []
for n in model_names:
    m = all_metrics[n]
    succ = m["success_rate"]
    sc   = m["avg_score"]
    tks  = m["avg_tokens"]

    # Means (keep as mean over per-difficulty means for stability/comparability)
    mean_success = float(np.mean(succ)) if succ else 0.0  # fraction [0,1]
    mean_score, _  = mean_std(sc)
    mean_tokens, _ = mean_std(tks)

    # SDs (switchable)
    if sd_mode == "all_samples":
        _, sd_score  = mean_std(m["score_samples"])
        _, sd_tokens = mean_std(m["token_samples"])
    else:  # "per_difficulty"
        _, sd_score  = mean_std(sc)
        _, sd_tokens = mean_std(tks)

    # Per-difficulty normalized metrics (mean only)
    norm_succ = [ (s/t)*SCALING if (t and t != 0) else np.nan for s, t in zip(succ, tks) ]
    norm_sc   = [ (x/t)*SCALING if (t and t != 0) else np.nan for x, t in zip(sc,   tks) ]
    mean_norm_succ, _ = mean_std(norm_succ)
    mean_norm_sc,   _ = mean_std(norm_sc)

    summary_rows.append((
        n,
        mean_success,            # as fraction; format as percent on output
        mean_score, sd_score,    # SD shown
        mean_tokens, sd_tokens,  # SD shown
        mean_norm_succ,          # no SD
        mean_norm_sc             # no SD
    ))

# sort by Mean Success (index 1)
summary_rows.sort(key=lambda x: x[1])

# ─── SAVE SUMMARY TO CSV ─────────────────────────────────────────────────────
csv_path = PLOTS_DIR / f"table_mean_{difficulty}.csv"
with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Model",
        "Mean Success (%)",
        "Mean Score", "Score SD",
        "Mean Tokens", "Tokens SD",
        f"Succ/{k_label}Tok",
        f"Sc/{k_label}Tok",
        "SD Mode"
    ])
    for (name, ms,
         msc, ssc,
         mt, st,
         mns, mnc) in summary_rows:
        writer.writerow([
            name,
            f"{ms * 100:.1f}",    # percent, one decimal
            f"{msc:.1f}",         # one decimal
            f"{ssc:.1f}",         # score SD one decimal
            f"{mt:.0f}",          # tokens mean: integer (0 decimals)
            f"{st:.0f}",          # tokens SD: integer (0 decimals)
            f"{mns:.2f}",         # normalized success per tokens*scale
            f"{mnc:.2f}",         # normalized score per tokens*scale
            sd_mode
        ])

print(f"\n[INFO] Summary table saved to: {csv_path}")

# ─── PRINT SUMMARY ───────────────────────────────────────────────────────────
print("Summary across difficulties:")
hdr = (
    f"{'Model':30}  "
    f"{'Mean Succ %':>12}  "
    f"{'Mean Sc ± SD':>15}  "
    f"{'Mean Tok ± SD':>16}  "
    f"{f'Succ/{k_label}Tok':>12}  "
    f"{f'Sc/{k_label}Tok':>12}  "
    f"{'SD Mode':>8}"
)
print(hdr)
print("-" * len(hdr))
for (name, ms,
     msc, ssc,
     mt, st,
     mns, mnc) in summary_rows:
    print(
        f"{name:30}  "
        f"{ms*100:12.1f}  "          # percent, one decimal
        f"{msc:6.1f} ± {ssc:4.1f}  " # mean score ± SD
        f"{mt:7.0f} ± {st:5.0f}  "   # tokens mean ± SD (0 decimals)
        f"{mns:12.2f}  "
        f"{mnc:12.2f}  "
        f"{sd_mode:>8}"
    )
