import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import os
import pickle
from pathlib import Path

PASTEL_COLORS = [
    "#82C9FF",  # blue
    "#FF8692",  # red
    "#4BDA6A",  # green
    "#DB97E3",  # purple
    "#FFC085",
    "#FFFF82",  # yellow
    "#7EDCD5"
]


# ==============================
# 1. SETTINGS
# ==============================

# Path to your result files
folder_path = "study_case_model/figures/volatility_stress/07052026"   # <-- change this
file_pattern = os.path.join(folder_path, "stress_test_results_*")

# ==============================
# 2. LOAD FILES + EXTRACT METADATA
# ==============================

files = glob.glob(file_pattern)

if not files:
    raise ValueError("No files found. Check your folder path or file pattern.")

all_data = []

pattern = re.compile(
    r"stress_test_results_(?P<subsidy>[-\d.]+)_(?P<corr>[-\d.]+)_price_shock_(?P<shock_type>lt|st)_(?P<run>\d+)"
)

for file in files:
    filename = os.path.basename(file)
    match = pattern.search(filename)

    if not match:
        print(f"Skipping file (pattern mismatch): {filename}")
        continue

    params = match.groupdict()

    try:
        df = pd.read_csv(file)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        continue

    # Add metadata columns
    df['subsidy'] = float(params['subsidy'])
    df['correlation'] = float(params['corr'])
    df['shock_type'] = params['shock_type']
    df['run'] = int(params['run'])

    all_data.append(df)

# Combine all data
combined_df = pd.concat(all_data, ignore_index=True)

print(f"\nLoaded {len(files)} files")
print(f"Total rows: {len(combined_df)}")

# ==============================
# 3. CHECK REQUIRED COLUMNS
# ==============================

required_cols = ['objective', 'avg_h2_production']
for col in required_cols:
    if col not in combined_df.columns:
        raise ValueError(f"Missing required column: {col}")

# ==============================
# 4. AVERAGES PER CHARACTERISTIC
# ==============================

for col in ['subsidy', 'correlation', 'shock_type', 'vol_multiplier']:
    print(f"\n{'='*60}")
    print(f"AVERAGES BY {col.upper()}")
    print(f"{'='*60}")

    summary = combined_df.groupby(col)[['objective', 'avg_h2_production']].mean()
    print(summary.round(4))

# ==============================
# 5. AVERAGES BY ALL CHARACTERISTICS
# ==============================

print(f"\n{'='*60}")
print("AVERAGES BY ALL CHARACTERISTICS")
print(f"{'='*60}")

summary_all = combined_df.groupby(
    ['subsidy', 'correlation', 'shock_type', 'vol_multiplier']
)[['objective', 'avg_h2_production']].mean()

print(summary_all.round(4))

# ==============================
# 6. OPTIONAL: INCLUDE STD DEV
# ==============================

print(f"\n{'='*60}")
print("MEAN + STD BY ALL CHARACTERISTICS")
print(f"{'='*60}")

summary_stats = combined_df.groupby(
    ['subsidy', 'correlation', 'shock_type']
)[['objective', 'avg_h2_production']].agg(['mean', 'std'])

print(summary_stats.round(4))

# ==============================
# 7. OPTIONAL: AVERAGE OVER RUNS
# ==============================

print(f"\n{'='*60}")
print("AVERAGED OVER RUNS (ROBUST RESULTS)")
print(f"{'='*60}")

# First average within each run (in case multiple rows per file)
per_run = combined_df.groupby(
    ['subsidy', 'correlation', 'shock_type', 'run']
)[['objective', 'avg_h2_production']].mean()

# Then average across runs
final_summary = per_run.groupby(
    ['subsidy', 'correlation', 'shock_type']
).mean()

print(final_summary.round(4))

# ==============================
# 8. OPTIONAL: SAVE RESULTS
# ==============================

# output_path = os.path.join(folder_path, "summary_results.csv")
# final_summary.to_csv(output_path)

# print(f"\nSaved final summary to: {output_path}")


# ==============================
# 9. CHECK REQUIRED COLUMNS FOR PLOTTING
# ==============================

required_plot_cols = ['vol_multiplier', 'cvar_5pct']
for col in required_plot_cols:
    if col not in combined_df.columns:
        raise ValueError(f"Missing required column for plotting: {col}")

# ==============================
# 10. PLOTTING FUNCTION
# ==============================
def plot_by_correlation_with_subsidy_lines(
    df, shock_type, y_col, title,
    folder="study_case_model/figures/volatility_stress/07052026/plots_by_correlation",
    aggregate_subsidies=False
):
    os.makedirs(folder, exist_ok=True)

    subset = df[df['shock_type'] == shock_type]

    plt.figure(figsize=(10, 5))
    global_min = 0

    if aggregate_subsidies:
        # --- ONE LINE PER CORRELATION ---
        for j, corr in enumerate(sorted(subset['correlation'].unique())):
            temp = subset[subset['correlation'] == corr]

            grouped = temp.groupby('vol_multiplier')[y_col]
            mean = grouped.mean()
            se = grouped.std() / np.sqrt(grouped.count())

            vol = mean.index.values
            diff = mean.values
            se_vals = se.values

            global_min = min(global_min, np.min(diff))

            color = PASTEL_COLORS[j % len(PASTEL_COLORS)]
            plt.plot(vol, diff, '-', label=f"corr={corr}", color=color)
            plt.fill_between(vol, diff - se_vals, diff + se_vals, color=color, alpha=0.2)

    else:
        # --- ONE LINE PER (corr, subsidy) ---
        grouped = subset.groupby(['correlation', 'subsidy'])

        for j, ((corr, subsidy), temp) in enumerate(grouped):
            grouped_inner = temp.groupby('vol_multiplier')[y_col]
            mean = grouped_inner.mean()
            se = grouped_inner.std() / np.sqrt(grouped_inner.count())

            vol = mean.index.values
            diff = mean.values
            se_vals = se.values

            global_min = min(global_min, np.min(diff))

            color = PASTEL_COLORS[j % len(PASTEL_COLORS)]
            plt.plot(vol, diff, '-', label=f"corr={corr}, sub={subsidy}", color=color)
            plt.fill_between(vol, diff - se_vals, diff + se_vals, color=color, alpha=0.2)

    plt.xlabel("Volatility Multiplier")
    plt.ylabel(f"{y_col}")
    plt.title(f"{title} | shock={shock_type}")

    plt.ylim(bottom=min(0, global_min))
    plt.grid(alpha=0.3, axis="y")
    plt.legend(loc="upper left")

    suffix = "agg_subsidy" if aggregate_subsidies else "full"
    filename = f"{y_col}_{suffix}_shock_{shock_type}.png"

    plt.savefig(os.path.join(folder, filename))
    plt.close()

def plot_by_subsidy_with_correlation_lines(df, shock_type, y_col, title, folder = "study_case_model/figures/volatility_stress/07052026/plots_by_subsidy"):
    os.makedirs(folder, exist_ok=True)

    subset = df[df['shock_type'] == shock_type]
    subsidies = sorted(subset['subsidy'].unique())

    for subsidy in subsidies:
        temp_sub = subset[subset['subsidy'] == subsidy]

        plt.figure(figsize=(10, 5))
        global_min = 0

        for j, corr in enumerate(sorted(temp_sub['correlation'].unique())):
            temp = temp_sub[temp_sub['correlation'] == corr]

            # # ---- BASELINE ----
            # baseline_df = subset[
            #     (subset['correlation'] == corr) &
            #     (subset['subsidy'] == 0) &
            #     (subset['vol_multiplier'] == 1)
            # ]

            # if baseline_df.empty:
            #     print(f"Skipping corr={corr}, subsidy={subsidy}, shock={shock_type}")
            #     continue

            # baseline = baseline_df[y_col].mean()

            grouped = temp.groupby('vol_multiplier')[y_col]
            mean = grouped.mean()
            se = grouped.std() / np.sqrt(grouped.count())

            vol = mean.index.values
            diff = mean.values # - baseline
            se_vals = se.values

            global_min = min(global_min, np.min(diff))

            color = PASTEL_COLORS[j % len(PASTEL_COLORS)]

            # Line
            plt.plot(vol, diff, '-', label=f"corr={corr}", color=color)

            # Shaded band
            lower = diff - se_vals
            upper = diff + se_vals
            plt.fill_between(vol, lower, upper, color=color, alpha=0.2)

        plt.xlabel("Volatility Multiplier")
        plt.ylabel(f"Δ {y_col} (vs subsidy=0, vol=1)")
        plt.title(f"{title} | subsidy={subsidy} | shock={shock_type}")

        plt.ylim(bottom=min(0, global_min))
        plt.grid(alpha=0.3, axis="y")
        plt.legend(loc="upper left")

        filename = f"{y_col}_subsidy_{subsidy}_shock_{shock_type}.png"
        plt.savefig(os.path.join(folder, filename))
        plt.close()


def compute_overprovision(snapshot):
    """
    Compute entry and exit overprovision values from a snapshot.

    Returns:
        entry_overprovisions (list), exit_overprovisions (list)
    """

    f_vals = snapshot["variables"]["f"]
    entry_vals = snapshot["variables"]["x_entry"]
    exit_vals = snapshot["variables"]["x_exit"]

    # =========================
    # Total capacity per node
    # =========================
    entry_overprovision = []
    exit_overprovision = []

    M_3 = set(s for (a_in, a_out, c, s) in f_vals.keys())
    K = [1,2,3]

    N = set(n for (n, k, s) in entry_vals.keys())

    for s in M_3:
        entry_overprovision_nodes = 0
        exit_overprovision_nodes = 0
        entry_amount = 0
        exit_amount = 0

        for n in N:

            entry_val = 0
            exit_val = 0

            for k in K:
                if k ==1:
                    s_prime =1
                elif k ==2:
                    s_prime = (s-1) // 8 +1
                else:
                    s_prime = s
                # average over scenarios of that stage
                entry_val += entry_vals[n, k, s_prime]
                exit_val += exit_vals[n, k, s_prime]

            inflow = sum(val for (a_in, a_out, c, s_prime), val in f_vals.items() if a_out == n and s_prime == s )
            outflow = sum(val for (a_in, a_out, c, s_prime), val in f_vals.items() if a_in == n and s_prime == s )


            net_flow = inflow - outflow
            # if abs(net_flow) > 1:
            #     print(n, net_flow)


            if net_flow >0.01:
                e_overprovision = entry_val
                ex_overprovision = max(exit_val - net_flow, 0)
                # print(n, e_overprovision)
                # print(n, ex_overprovision)
            elif net_flow < -0.01:
                e_overprovision = max(entry_val + net_flow, 0)
                ex_overprovision = exit_val
                # print(n, e_overprovision)
                # print(n, ex_overprovision)
            else:
                e_overprovision = 0
                ex_overprovision = 0
                entry_val =0 
                ex_val = 0

            
            entry_overprovision_nodes += e_overprovision
            exit_overprovision_nodes += ex_overprovision
            entry_amount += entry_val
            exit_amount += exit_val
        
        entry_overprovision.append(entry_overprovision_nodes/ entry_amount if entry_amount > 0 else 0)
        exit_overprovision.append(exit_overprovision_nodes/ exit_amount if exit_amount > 0 else 0)

    return entry_overprovision, exit_overprovision

def analyze_overprovision(base_folder, vol_multiplier, correlation, shock_type, subsidy):
    base = Path(base_folder)

    sub_dir = (
    base
    / f"vol{vol_multiplier}_{correlation}_price_{shock_type}"
    / f"sub{subsidy}"
    )
    print(sub_dir)
    entry_overprovisions = []
    exit_overprovisions = []

    for run_dir in sorted(sub_dir.glob("run*/")):
        files = list(run_dir.glob("*.pkl"))
        if not files:
            continue

        with open(files[0], "rb") as f:
            snapshot = pickle.load(f)
            entry_overprovision, exit_overprovision = compute_overprovision(snapshot)
            entry_overprovisions.append(np.mean(entry_overprovision))
            exit_overprovisions.append(np.mean(exit_overprovision))

    return entry_overprovisions, exit_overprovisions


def build_overprovision_df(base_folder):
    records = []

    for vol_dir in Path(base_folder).glob("vol*/"):
        vol_string = vol_dir.name.replace("vol", "")
        parameters_string = vol_string.replace("price_", "")
        volatility, correlation, shock_type = [param for param in parameters_string.split("_")]
        volatility = float(volatility)
        correlation = float(correlation)

        for sub_dir in vol_dir.glob("sub*/"):
            subsidy = float(sub_dir.name.replace("sub", ""))

            entry_vals, exit_vals = analyze_overprovision(base_folder, volatility, correlation, shock_type, subsidy)

            if len(entry_vals) == 0:
                continue

            for i, values in enumerate(zip(entry_vals, exit_vals)):
                records.append({
                    "vol_multiplier": volatility,
                    "correlation": correlation,
                    "shock_type" : shock_type,
                    "subsidy": subsidy,
                    "run": i, 
                    "entry_overprov": values[0],
                    "exit_overprov": values[1]
                })

    return pd.DataFrame(records)

# ==============================
# OVERPROVISION DATA
# ==============================

# overprov_base = "study_case_model/figures/volatility_stress/07052026"  # same base

# overprov_df = build_overprovision_df(overprov_base)

# # Merge on subsidy + deviation (vol_multiplier ≈ deviation proxy)
# combined_df = combined_df.merge(
#     overprov_df,
#     how="left",
#     left_on=["subsidy", "vol_multiplier", "correlation", "shock_type", "run"],
#     right_on=["subsidy", "vol_multiplier", "correlation", "shock_type", "run"]
# )


# ==============================
# GENERATE PLOTS
# ==============================

for shock in ['lt', 'st']:

    # ---- Objective ----
    plot_by_correlation_with_subsidy_lines(
        combined_df,
        shock,
        'objective',
        "Objective"
    )

    # ---- CVaR ----
    plot_by_correlation_with_subsidy_lines(
        combined_df,
        shock,
        'cvar_5pct',
        "CVaR (5%)"
    )

    # ---- Hydrogen ----
    plot_by_correlation_with_subsidy_lines(
        combined_df,
        shock,
        'avg_h2_production',
        "Hydrogen Production"
    )

    plot_by_correlation_with_subsidy_lines(
        combined_df,
        shock, 
        'std_h2_production',
        "Hydrogen Production (std dev)"
    )

    # plot_by_correlation_with_subsidy_lines(
    #     combined_df,
    #     shock, 
    #     "entry_overprov",
    #     "Total Entry Overprovision"
    # )

    # plot_by_correlation_with_subsidy_lines(
    #     combined_df,
    #     shock, 
    #     "exit_overprov",
    #     "Total Exit Overprovision"
    # )


    plot_by_subsidy_with_correlation_lines(
        combined_df,
        shock,
        'objective',
        "Objective"
    )

    plot_by_subsidy_with_correlation_lines(
        combined_df,
        shock,
        'cvar_5pct',
        "CVaR (5%)"
    )

    plot_by_subsidy_with_correlation_lines(
        combined_df,
        shock,
        'avg_h2_production',
        "Hydrogen Production"
    )

    plot_by_subsidy_with_correlation_lines(
        combined_df,
        shock, 
        'std_h2_production',
        "Hydrogen Production (std dev)"
    )

    # plot_by_subsidy_with_correlation_lines(
    #     combined_df, 
    #     shock,
    #     "entry_overprov",
    #     "Total Entry Overprovision"
    # )

    # plot_by_subsidy_with_correlation_lines(
    #     combined_df, 
    #     shock,
    #     "exit_overprov",
    #     "Total Exit Overprovision"
    # )
    

    