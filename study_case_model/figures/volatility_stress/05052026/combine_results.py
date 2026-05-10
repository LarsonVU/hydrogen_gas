import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import os

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
folder_path = "study_case_model/figures/volatility_stress/05052026"   # <-- change this
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
def plot_by_correlation_with_subsidy_lines(df, shock_type, y_col, title, folder = "study_case_model/figures/volatility_stress/05052026/plots_by_correlation"):
    os.makedirs(folder, exist_ok=True)

    subset = df[df['shock_type'] == shock_type]
    correlations = sorted(subset['correlation'].unique())

    for corr in correlations:
        temp_corr = subset[subset['correlation'] == corr]

        # ---- GLOBAL BASELINE ----
        baseline_df = temp_corr[
            (temp_corr['subsidy'] == 0) &
            (temp_corr['vol_multiplier'] == 1)
        ]

        if baseline_df.empty:
            print(f"Skipping corr={corr}, shock={shock_type} (no baseline)")
            continue

        baseline = baseline_df[y_col].mean()

        plt.figure(figsize=(10, 5))
        global_min = 0

        for j, subsidy in enumerate(sorted(temp_corr['subsidy'].unique())):
            temp = temp_corr[temp_corr['subsidy'] == subsidy]

            grouped = temp.groupby('vol_multiplier')[y_col]
            mean = grouped.mean()
            se = grouped.std() / np.sqrt(grouped.count())

            vol = mean.index.values
            diff = mean.values - baseline
            se_vals = se.values

            global_min = min(global_min, np.min(diff))

            color = PASTEL_COLORS[j % len(PASTEL_COLORS)]

            # Line
            plt.plot(vol, diff, '-', label=f"Subsidy {subsidy}", color=color)

            # Shaded band
            lower = diff - se_vals
            upper = diff + se_vals
            plt.fill_between(vol, lower, upper, color=color, alpha=0.2)

        plt.xlabel("Volatility Multiplier")
        plt.ylabel(f"Δ {y_col} (vs subsidy=0, vol=1)")
        plt.title(f"{title} | corr={corr} | shock={shock_type}")

        plt.ylim(bottom=min(0, global_min))
        plt.grid(alpha=0.3, axis="y")
        plt.legend(loc="upper left")

        filename = f"{y_col}_corr_{corr}_shock_{shock_type}.png"
        plt.savefig(os.path.join(folder, filename))
        plt.close()


def plot_by_subsidy_with_correlation_lines(df, shock_type, y_col, title, folder = "study_case_model/figures/volatility_stress/05052026/plots_by_subsidy"):
    os.makedirs(folder, exist_ok=True)

    subset = df[df['shock_type'] == shock_type]
    subsidies = sorted(subset['subsidy'].unique())

    for subsidy in subsidies:
        temp_sub = subset[subset['subsidy'] == subsidy]

        plt.figure(figsize=(10, 5))
        global_min = 0

        for j, corr in enumerate(sorted(temp_sub['correlation'].unique())):
            temp = temp_sub[temp_sub['correlation'] == corr]

            # ---- BASELINE ----
            baseline_df = subset[
                (subset['correlation'] == corr) &
                (subset['subsidy'] == 0) &
                (subset['vol_multiplier'] == 1)
            ]

            if baseline_df.empty:
                print(f"Skipping corr={corr}, subsidy={subsidy}, shock={shock_type}")
                continue

            baseline = baseline_df[y_col].mean()

            grouped = temp.groupby('vol_multiplier')[y_col]
            mean = grouped.mean()
            se = grouped.std() / np.sqrt(grouped.count())

            vol = mean.index.values
            diff = mean.values - baseline
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

    plot_by_subsidy_with_correlation_lines(
        combined_df, 
        shock,
        "total_entry_booking",
        "Total Entry Booking"
    )

    plot_by_subsidy_with_correlation_lines(
        combined_df, 
        shock,
        "total_exit_booking",
        "Total Exit Booking"
    )
    