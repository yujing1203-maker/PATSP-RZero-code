import os, glob, json
import numpy as np
import pandas as pd

stats_dir = "results/results_base_scoring"
out_dir   = "results/plots_tables"
os.makedirs(out_dir, exist_ok=True)

TOTAL_TASKS_DEFAULT = 180      # total across all splits
SPLIT_TASKS_DEFAULT = 20       # default per split (keys "1".."9")

def extract_metrics(total):
    m = {}
    ee = total.get("execution_errors", {})
    m["execution_errors_total"] = ee.get("total", 0)

    for func, cnt in ee.get("by_func", {}).items():
        m[f"exec_error_by_func_{func}"] = cnt
    for reason, cnt in ee.get("craft_reasons", {}).items():
        m[f"exec_error_craft_reason_{reason}"] = cnt

    m["craft_missing_not_attempted"] = total.get(
        "craft_missing_not_attempted",
        total.get("craft-missing", {}).get("craft_missing_not_attempted", 0)
    )
    m["craft_missing_attempted"] = total.get(
        "craft_missing_attempted",
        total.get("craft-missing", {}).get("craft_missing_attempted", 0)
    )
    for reason, cnt in total.get("craft-missing", {}).get("reasons", {}).items():
        m[f"craft_missing_reason_{reason}"] = cnt

    m["craft_intermediate_not_attempted"] = total.get(
        "craft_intermediate_not_attempted",
        total.get("craft-intermediate", {}).get("craft_intermediate_not_attempted", 0)
    )
    m["craft_intermediate_attempted"] = total.get(
        "craft_intermediate_attempted",
        total.get("craft-intermediate", {}).get("craft_intermediate_attempted", 0)
    )
    for reason, cnt in total.get("craft-intermediate", {}).get("reasons", {}).items():
        m[f"craft_intermediate_reason_{reason}"] = cnt

    for key in (
        "tasks_with_only_failed_gear_calculation",
        "tasks_failed_total",
        "tasks_with_both_calc_craft",
        "tasks_with_only_failed_craft",
        "tasks_with_wrong_code_format",
        "tasks_other",
        "win_tasks",
        "num_tasks_total",
    ):
        m[key] = total.get(key, 0)
    return m

def as_int(x):
    try:
        return int(round(float(x)))
    except Exception:
        return 0

def to_count_rate_like(x, total):
    """Accepts: 0.028, 2.8, '12.8%', or a count."""
    if isinstance(x, str):
        s = x.strip()
        if s.endswith("%"):
            try:
                return as_int(float(s[:-1]) / 100.0 * total)
            except Exception:
                return 0
        try:
            x = float(s)
        except Exception:
            return 0
    v = float(x)
    if v <= 1.0:
        return as_int(v * total)                 # fraction
    if 1.0 < v <= 100.0 and abs(v - round(v)) > 1e-9:
        return as_int((v / 100.0) * total)      # percent like 12.8
    return as_int(v)                             # already a count

def compute_sd_across_splits(data_dict):
    """
    Compute SD across keys '1'..'9' (if present), using per-eligible-task means PER SPLIT.
    Returns (craft_sd, exec_sd) rounded to 2 decimals.
    """
    craft_means, exec_means = [], []

    for k in map(str, range(1, 10)):
        block = data_dict.get(k)
        if not isinstance(block, dict):
            continue

        met = extract_metrics(block)
        # denominator for this split
        split_total = met.get("num_tasks_total", 0)
        split_total = as_int(split_total) if as_int(split_total) > 0 else SPLIT_TASKS_DEFAULT
        wrong_split = to_count_rate_like(met.get("tasks_with_wrong_code_format", 0), split_total)
        eligible_split = max(split_total - wrong_split, 0)
        if eligible_split <= 0:
            continue

        craft_mean = float(met["craft_missing_not_attempted"]) / eligible_split
        exec_mean  = float(met["execution_errors_total"]) / eligible_split
        craft_means.append(craft_mean)
        exec_means.append(exec_mean)

    if len(craft_means) > 1:
        craft_sd = float(np.round(np.std(craft_means, ddof=1), 2))
    else:
        craft_sd = 0.00
    if len(exec_means) > 1:
        exec_sd = float(np.round(np.std(exec_means, ddof=1), 2))
    else:
        exec_sd = 0.00
    return craft_sd, exec_sd

# ----- load -----
rows = []
sd_by_model = {}
for fn in glob.glob(os.path.join(stats_dir, "*_stats.json")):
    model = os.path.basename(fn).replace("_stats.json", "")
    with open(fn, "r") as f:
        data = json.load(f)

    metrics = extract_metrics(data.get("total", {}))
    metrics["model"] = model
    rows.append(metrics)

    # SD across split keys 1..9
    c_sd, e_sd = compute_sd_across_splits(data)
    sd_by_model[model] = (c_sd, e_sd)

df = pd.DataFrame(rows).set_index("model").fillna(0)

# ----- stable denominator per model -----
if "num_tasks_total" in df.columns:
    total_tasks = df["num_tasks_total"].apply(lambda x: as_int(x) if as_int(x) > 0 else TOTAL_TASKS_DEFAULT)
else:
    total_tasks = pd.Series(TOTAL_TASKS_DEFAULT, index=df.index)

wrong_code_count = pd.Series(index=df.index, dtype=int)
for m in df.index:
    wrong_code_count[m] = to_count_rate_like(df.at[m, "tasks_with_wrong_code_format"], total_tasks[m])

eligible = (total_tasks - wrong_code_count).clip(lower=0).astype(int)

# ----- per-eligible-task means (overall) -----
df["craft_missing_not_attempted"] = (
    df["craft_missing_not_attempted"] / eligible.replace(0, np.nan)
).fillna(0.0).round(3)

df["execution_errors_total"] = (
    df["execution_errors_total"] / eligible.replace(0, np.nan)
).fillna(0.0).round(3)

# ----- attach SDs computed across splits -----
df["craft_missing_not_attempted_sd"] = [sd_by_model[m][0] for m in df.index]
df["execution_errors_total_sd"]      = [sd_by_model[m][1] for m in df.index]

# optional audit columns
df["eligible_tasks"] = eligible
df["wrong_code_as_count"] = wrong_code_count
df["total_tasks_used"] = total_tasks

# ----- save -----
df.to_csv(os.path.join(out_dir, "models_comparison1.csv"))

columns_to_keep = [
    "craft_missing_not_attempted",        # mean per eligible task (overall)
    "craft_missing_not_attempted_sd",     # SD across splits 1..9
    "execution_errors_total",             # mean per eligible task (overall)
    "execution_errors_total_sd",          # SD across splits 1..9
    "tasks_failed_total",
    "tasks_with_only_failed_gear_calculation",
    "tasks_with_both_calc_craft",
    "tasks_with_only_failed_craft",
    "tasks_with_wrong_code_format",
    "wrong_code_as_count",
    "eligible_tasks",
    "total_tasks_used",
]
df_subset = df.reindex(columns=columns_to_keep, fill_value=0)
df_subset.to_csv(os.path.join(out_dir, "models_comparison_subset1.csv"))
print(df_subset)
