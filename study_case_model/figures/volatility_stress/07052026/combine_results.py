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
HYDROGEN_MSCM_GWH = 2.78  #LHV in GWh per MSCM of hydrogen

# ==============================
# 1. SETTINGS
# ==============================


# ==============================
# 2. LOAD FILES + EXTRACT METADATA
# ==============================
def collect_and_combine_results(folder_path, file_pattern):
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

    return combined_df

def compute_stage_capacity_fractions(snapshot):
    """
    Compute the expected fraction of entry and exit capacity bought in each stage.

    Returns:
        entry_stage_fraction (list): length-3 fractions for stages 1, 2, 3
        exit_stage_fraction (list): length-3 fractions for stages 1, 2, 3
    """

    entry_vals = snapshot["variables"]["x_entry"]
    exit_vals = snapshot["variables"]["x_exit"]

    # nodes present in entry decisions
    nodes = set(n for (n, k, s) in entry_vals.keys())
    # actual stage-3 scenario indices
    stage3_scenarios = set(s for (n, k, s) in entry_vals.keys() if k == 3)

    if not stage3_scenarios:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    entry_stage_totals = {1: 0.0, 2: 0.0, 3: 0.0}
    exit_stage_totals = {1: 0.0, 2: 0.0, 3: 0.0}

    for s in stage3_scenarios:
        entry_stage_values = {1: 0.0, 2: 0.0, 3: 0.0}
        exit_stage_values = {1: 0.0, 2: 0.0, 3: 0.0}

        scenario_group = (s - 1) // 8 + 1

        for n in nodes:
            entry_stage_values[1] += entry_vals.get((n, 1, 1), 0.0)
            entry_stage_values[2] += entry_vals.get((n, 2, scenario_group), 0.0)
            entry_stage_values[3] += entry_vals.get((n, 3, s), 0.0)

            exit_stage_values[1] += exit_vals.get((n, 1, 1), 0.0)
            exit_stage_values[2] += exit_vals.get((n, 2, scenario_group), 0.0)
            exit_stage_values[3] += exit_vals.get((n, 3, s), 0.0)

        for k in [1, 2, 3]:
            entry_stage_totals[k] += entry_stage_values[k]
            exit_stage_totals[k] += exit_stage_values[k]

    scenario_count = len(stage3_scenarios)
    for k in [1, 2, 3]:
        entry_stage_totals[k] /= scenario_count
        exit_stage_totals[k] /= scenario_count

    total_entry = sum(entry_stage_totals.values())
    total_exit = sum(exit_stage_totals.values())

    entry_stage_fraction = [
        entry_stage_totals[k] / total_entry if total_entry > 0 else 0.0
        for k in [1, 2, 3]
    ]
    exit_stage_fraction = [
        exit_stage_totals[k] / total_exit if total_exit > 0 else 0.0
        for k in [1, 2, 3]
    ]

    return entry_stage_fraction, exit_stage_fraction


def analyze_stage_capacity(base_folder, vol_multiplier, correlation, shock_type, subsidy):
    base = Path(base_folder)

    sub_dir = (
        base
        / f"vol{vol_multiplier}_{correlation}_price_{shock_type}"
        / f"sub{subsidy}"
    )
    print(sub_dir)
    entry_stage_fractions = []
    exit_stage_fractions = []

    for run_dir in sorted(sub_dir.glob("run*/")):
        files = list(run_dir.glob("*.pkl"))
        if not files:
            continue

        with open(files[0], "rb") as f:
            snapshot = pickle.load(f)
            entry_fraction, exit_fraction = compute_stage_capacity_fractions(snapshot)
            entry_stage_fractions.append(entry_fraction)
            exit_stage_fractions.append(exit_fraction)

    return entry_stage_fractions, exit_stage_fractions


def build_stage_capacity_df(base_folder):
    records = []

    for vol_dir in Path(base_folder).glob("vol*/"):
        vol_string = vol_dir.name.replace("vol", "")
        parameters_string = vol_string.replace("price_", "")
        volatility, correlation, shock_type = [param for param in parameters_string.split("_")]
        volatility = float(volatility)
        correlation = float(correlation)

        for sub_dir in vol_dir.glob("sub*/"):
            subsidy = float(sub_dir.name.replace("sub", ""))

            entry_fracs, exit_fracs = analyze_stage_capacity(base_folder, volatility, correlation, shock_type, subsidy)

            if len(entry_fracs) == 0:
                continue

            for i, (entry_frac, exit_frac) in enumerate(zip(entry_fracs, exit_fracs)):
                records.append({
                    "vol_multiplier": volatility,
                    "correlation": correlation,
                    "shock_type": shock_type,
                    "subsidy": subsidy,
                    "run": i,
                    "entry_frac_stage_1": entry_frac[0],
                    "entry_frac_stage_2": entry_frac[1],
                    "entry_frac_stage_3": entry_frac[2],
                    "exit_frac_stage_1": exit_frac[0],
                    "exit_frac_stage_2": exit_frac[1],
                    "exit_frac_stage_3": exit_frac[2],
                })

    return pd.DataFrame(records)

# ==============================
# OVERPROVISION DATA
# ==============================

# overprov_base = "study_case_model/figures/volatility_stress/07052026"  # same base

# overprov_df = build_overprovision_df(overprov_base)

# stage_capacity_df = build_stage_capacity_df(overprov_base)

# combined_df = combined_df.merge(
#     overprov_df,
#     how="left",
#     left_on=["subsidy", "vol_multiplier", "correlation", "shock_type", "run"],
#     right_on=["subsidy", "vol_multiplier", "correlation", "shock_type", "run"]
# )



def add_stage_capacity_fractions(df, base_folder):
    stage_capacity_df = build_stage_capacity_df(base_folder)

    combined_df = stage_capacity_df.merge(
        df,
        how="left",
        left_on=["subsidy", "vol_multiplier", "correlation", "shock_type", "run"],
        right_on=["subsidy", "vol_multiplier", "correlation", "shock_type", "run"]
    )
    return combined_df

# ==============================
# 10. PLOTTING FUNCTION
# ==============================
def plot_cvar(
    df, shock_type,
    folder="study_case_model/figures/volatility_stress/07052026/plots_by_correlation",
):
    os.makedirs(folder, exist_ok=True)

    subset = df[df['shock_type'] == shock_type]

    plt.figure(figsize=(10, 5))
    global_min = 0

    # baseline: vol_multiplier == 1 and correlation == 0.0 (average over subsidies/runs)
    base_mask = (subset['vol_multiplier'] == 1) & (subset['correlation'] == 0.0)
    if base_mask.any():
        baseline = subset.loc[base_mask, 'cvar_5pct'].mean() / 1000000
    else:
        baseline = 0.0

    # ---- COMBINE ALL SUBSIDY RUNS ----
    for j, corr in enumerate(sorted(subset['correlation'].unique())):

        temp = subset[subset['correlation'] == corr]

        # Use CRN: compute per-run averages (over subsidies), then subtract per-run baseline
        # grouped per vol_multiplier and run
        per_run = temp.groupby(['vol_multiplier', 'run'])['cvar_5pct'].mean()
        # pivot to have vol_multiplier as rows and run as columns
        pivot = per_run.unstack(level='run')  # rows: vol_multiplier, cols: run

        # baseline per run: from vol_multiplier==1 & correlation==0.0 in the whole subset
        base_mask_run = (subset['vol_multiplier'] == 1) & (subset['correlation'] == 0.0)
        if base_mask_run.any():
            base_per_run = subset.loc[base_mask_run].groupby('run')['cvar_5pct'].mean()
        else:
            # fallback to zeros if no per-run baseline available
            base_per_run = pd.Series(0.0, index=pivot.columns)

        # align baseline columns with pivot columns (runs)
        base_per_run = base_per_run.reindex(pivot.columns, fill_value=0.0)

        # subtract baseline per run (CRN)
        pivot_diff = pivot.subtract(base_per_run, axis=1)

        # compute mean and standard error across runs AFTER differencing
        mean = pivot_diff.mean(axis=1)
        se = pivot_diff.std(axis=1) / np.sqrt(pivot_diff.count(axis=1))

        vol = mean.index.values
        vals = mean.values / 1000000
        se_vals = se.values / 1000000

        global_min = min(global_min, np.min(vals))

        color = PASTEL_COLORS[j % len(PASTEL_COLORS)]

        plt.plot(
            vol,
            vals,
            '-',
            label=r"$r = {:.0f}\%$".format(corr * 100),
            color=color
        )

        plt.fill_between(
            vol,
            vals - se_vals,
            vals + se_vals,
            color=color,
            alpha=0.2
        )

    plt.xlabel("Volatility Multiplier")
    plt.ylabel("Change in 5%CVaR (Million Euro)")
    if shock_type == "lt":
        plt.title(f"Change in 5% Cvar for Long-term Price Shock")
    elif shock_type == "st":
        plt.title(f"Change in 5% Cvar for Day Ahead Price Shock")

    #plt.ylim(bottom=global_min - 0.1 * abs(global_min))
    plt.grid(alpha=0.3, axis="y")
    plt.legend(loc="lower left")

    filename = f"cvar_combined_subsidies_shock_{shock_type}.png"

    plt.savefig(os.path.join(folder, filename))
    plt.close()


def plot_objective(
    df, shock_type,
    folder="study_case_model/figures/volatility_stress/07052026/plots_by_correlation",
):
    os.makedirs(folder, exist_ok=True)

    subset = df[df['shock_type'] == shock_type]

    plt.figure(figsize=(10, 5))
    global_min = 0

    # ---- COMBINE ALL SUBSIDY RUNS ----
    for j, corr in enumerate(sorted(subset['correlation'].unique())):

        temp = subset[subset['correlation'] == corr]

        # Use CRN: compute per-run averages (over subsidies), then subtract per-run baseline
        per_run = temp.groupby(['vol_multiplier', 'run'])['objective'].mean()
        pivot = per_run.unstack(level='run')  # rows: vol_multiplier, cols: run

        # baseline per run: from vol_multiplier==1 & correlation==0.0 in the whole subset
        base_mask_run = (subset['vol_multiplier'] == 1) & (subset['correlation'] == 0.0)
        if base_mask_run.any():
            base_per_run = subset.loc[base_mask_run].groupby('run')['objective'].mean()
        else:
            base_per_run = pd.Series(0.0, index=pivot.columns)

        base_per_run = base_per_run.reindex(pivot.columns, fill_value=0.0)

        pivot_diff = pivot.subtract(base_per_run, axis=1)

        mean = pivot_diff.mean(axis=1)
        se = pivot_diff.std(axis=1) / np.sqrt(pivot_diff.count(axis=1))

        vol = mean.index.values
        vals = mean.values / 1000000
        se_vals = se.values / 1000000

        global_min = min(global_min, np.min(vals))

        color = PASTEL_COLORS[j % len(PASTEL_COLORS)]

        plt.plot(
            vol,
            vals,
            '-',
            label=r"$r = {:.0f}\%$".format(corr * 100),
            color=color
        )

        plt.fill_between(
            vol,
            vals - se_vals,
            vals + se_vals,
            color=color,
            alpha=0.2
        )

    plt.xlabel("Volatility Multiplier")
    plt.ylabel("Change in Objective Value (Million Euro)")
    if shock_type == "lt":
        plt.title(f"Change in Objective Value for Long-term Price Shock")
    elif shock_type == "st":
        plt.title(f"Change in Objective Value for Day Ahead Price Shock")

    #plt.ylim(bottom=global_min - 0.1 * abs(global_min))
    plt.grid(alpha=0.3, axis="y")
    plt.legend(loc="upper left")

    filename = f"objective_combined_subsidies_shock_{shock_type}.png"

    plt.savefig(os.path.join(folder, filename))
    plt.close()


def plot_stage_capacity_fractions_combined(
    df,
    folder="study_case_model/figures/volatility_stress/07052026/plots_stage_capacity",
):
    os.makedirs(folder, exist_ok=True)

    configs = [
        ("lt", "entry", "Long-term\nEntry Capacity"),
        ("st", "entry", "Day-ahead\nEntry Capacity"),
        ("lt", "exit", "Long-term\nExit Capacity"),
        ("st", "exit", "Day-ahead\nExit Capacity"),
    ]

    stage_labels = ["Stage 1", "Stage 2", "Stage 3"]

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(10, 8),
        sharex=True,
    )

    for ax, (shock_type, capacity_type, side_title) in zip(axes, configs):

        subset = df[df['shock_type'] == shock_type]

        if capacity_type == "entry":
            cols = [
                "entry_frac_stage_1",
                "entry_frac_stage_2",
                "entry_frac_stage_3",
            ]
        else:
            cols = [
                "exit_frac_stage_1",
                "exit_frac_stage_2",
                "exit_frac_stage_3",
            ]

        experiments = subset.groupby([
            "vol_multiplier",
            "correlation",
            "subsidy",
            "run",
        ])[cols].mean()

        grouped = experiments.groupby("vol_multiplier")
        mean = grouped.mean()
        se = grouped.std(ddof=1) / np.sqrt(grouped.count())

        for idx, col in enumerate(cols):

            vol = mean.index.values
            vals = mean[col].values * 100
            se_vals = se[col].values * 100

            color = PASTEL_COLORS[idx % len(PASTEL_COLORS)]

            ax.plot(
                vol,
                vals,
                '-',
                label=stage_labels[idx],
                color=color,
                linewidth=2,
            )

            ax.fill_between(
                vol,
                vals - se_vals,
                vals + se_vals,
                color=color,
                alpha=0.2,
            )

        # Individual subplot formatting
        ax.set_ylim(-2, 102)
        ax.grid(alpha=0.3, axis="y")

        # Left-side subplot title
        ax.text(
            -0.12,
            0.5,
            side_title,
            transform=ax.transAxes,
            rotation=0,
            va='center',
            ha='right',
            fontsize=12,
            fontweight='bold',
        )

        # Individual y-axis
        ax.set_ylabel("Capacity (%)")

    # Shared x-axis
    axes[-1].set_xlabel("Volatility Multiplier")

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()

    # Move legend and title slightly downward
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.58, 0.925),
    )

    fig.suptitle(
        "Average Capacity Allocation Fractions by Stage",
        fontsize=16,
        fontweight='bold',
        x=0.58,
        y=0.955,
    )

    # Increase upper bound in rect so subplot area starts higher
    plt.tight_layout(
        rect=[0.12, 0.04, 1, 0.915]
    )

    filename = "combined_stage_capacity_fractions.png"
    plt.savefig(
        os.path.join(folder, filename),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_stage_capacities_fraction_correlation(df, 
                                               shock_type, 
                                               capacity_type, 
                                               stage,
                                               folder = "study_case_model/figures/volatility_stress/07052026/plots_stage_capacity"):
    os.makedirs(folder, exist_ok=True)

    subset = df[df['shock_type'] == shock_type]

    plt.figure(figsize=(10, 5))
    global_min = 0

    # ---- COMBINE ALL SUBSIDY RUNS ----
    for j, corr in enumerate(sorted(subset['correlation'].unique())):

        temp = subset[subset['correlation'] == corr]

        # Use CRN: compute per-run averages (over subsidies), then subtract per-run baseline
        per_run = temp.groupby(['vol_multiplier', 'run'])[f'{capacity_type}_frac_stage_{stage}'].mean()
        pivot = per_run.unstack(level='run')  # rows: vol_multiplier, cols: run

        mean = pivot.mean(axis=1)
        se = pivot.std(axis=1) / np.sqrt(pivot.count(axis=1))

        vol = mean.index.values 
        vals = mean.values * 100
        se_vals = se.values * 100

        global_min = min(global_min, np.min(vals))

        color = PASTEL_COLORS[j % len(PASTEL_COLORS)]

        plt.plot(
            vol,
            vals,
            '-',
            label=r"$r = {:.0f}\%$".format(corr * 100),
            color=color
        )

        plt.fill_between(
            vol,
            vals - se_vals,
            vals + se_vals,
            color=color,
            alpha=0.2
        )

    plt.xlabel("Volatility Multiplier")
    plt.ylabel(f"{capacity_type.capitalize()} Capacity (%)")
    if shock_type == "lt":
        plt.title(f"Bought {capacity_type.capitalize()} Capacity Fraction for Long-term Price Shock in Stage {stage}")
    elif shock_type == "st":
        plt.title(f"Bought {capacity_type.capitalize()} Capacity Fraction for Day Ahead Price Shock in Stage {stage}")

    #plt.ylim(bottom=global_min - 0.1 * abs(global_min))
    plt.grid(alpha=0.3, axis="y")
    plt.legend(loc="upper left")

    filename = f"{capacity_type}_fraction_{stage}_combined_subsidies_shock_{shock_type}.png"

    plt.savefig(os.path.join(folder, filename))
    plt.close()

def plot_avg_hydrogen_production(
    df,
    shock_type,
    folder="study_case_model/figures/volatility_stress/07052026/plots_stage_hyd_production",
):
    os.makedirs(folder, exist_ok=True)

    subset = df[df['shock_type'] == shock_type]

    if subset.empty:
        print(f"No data for shock_type {shock_type}")
        return

    col = "avg_h2_production"

    # Average per experiment
    experiments = subset.groupby([
        "vol_multiplier",
        "correlation",
        "subsidy",
        "run",
    ])[col].mean().reset_index()

    # Aggregate over runs/correlations
    grouped = experiments.groupby(["vol_multiplier", "subsidy"])[col]

    mean = grouped.mean().reset_index(name="mean")
    se = (
        grouped.std(ddof=1) / np.sqrt(grouped.count())
    ).reset_index(name="se") 

    plot_df = pd.merge(mean, se, on=["vol_multiplier", "subsidy"])

    subsidy_levels = sorted(plot_df["subsidy"].unique())

    plt.figure(figsize=(10, 5))

    for idx, subsidy in enumerate(subsidy_levels):

        sub_df = plot_df[plot_df["subsidy"] == subsidy].sort_values("vol_multiplier")

        vol = sub_df["vol_multiplier"].values
        vals = sub_df["mean"].values  * HYDROGEN_MSCM_GWH
        se_vals = sub_df["se"].values * HYDROGEN_MSCM_GWH

        color = PASTEL_COLORS[idx % len(PASTEL_COLORS)]

        plt.plot(
            vol,
            vals,
            '-',
            label=f"Subsidy {int(subsidy)} €/MWh",
            color=color,
        )

        plt.fill_between(
            vol,
            vals - se_vals,
            vals + se_vals,
            color=color,
            alpha=0.2,
        )

    plt.xlabel("Volatility Multiplier")
    plt.ylabel("Average H2 Production (GWh)")

    shock_label = "Long-term" if shock_type == "lt" else "Day Ahead"

    plt.title(
        f"Average Hydrogen Production for Different Subsidy Levels ({shock_label} Price Shock)"
    )

    plt.grid(alpha=0.3, axis="y")
    plt.legend(loc="lower right", fontsize="small")

    filename = f"avg_h2_production_shock_{shock_type}.png"

    plt.savefig(os.path.join(folder, filename), bbox_inches="tight")
    plt.close()


def plot_avg_hydrogen_production_vs_correlation(
    df,
    shock_type,
    vol_multiplier=None,
    folder="study_case_model/figures/volatility_stress/07052026/plots_stage_hyd_production",
):
    os.makedirs(folder, exist_ok=True)

    subset = df[df['shock_type'] == shock_type]
    if vol_multiplier is not None:
        subset = subset[subset['vol_multiplier'] == vol_multiplier]
        if subset.empty:
            print(f"No data for shock_type {shock_type} and vol_multiplier {vol_multiplier}")
            return

    if subset.empty:
        print(f"No data for shock_type {shock_type}")
        return

    col = "avg_h2_production"

    experiments = subset.groupby([
        "correlation",
        "subsidy",
        "run",
    ])[col].mean().reset_index()

    grouped = experiments.groupby([
        "correlation",
        "subsidy",
    ])[col]

    mean = grouped.mean().reset_index(name="mean")
    se = (
        grouped.std(ddof=1) / np.sqrt(grouped.count())
    ).reset_index(name="se").fillna(0)

    plot_df = pd.merge(mean, se, on=["correlation", "subsidy"])

    subsidy_levels = sorted(plot_df["subsidy"].unique())

    plt.figure(figsize=(10, 5))

    for idx, subsidy in enumerate(subsidy_levels):
        sub_df = plot_df[plot_df["subsidy"] == subsidy].sort_values("correlation")

        corr = sub_df["correlation"].values
        vals = sub_df["mean"].values * HYDROGEN_MSCM_GWH
        se_vals = sub_df["se"].values * HYDROGEN_MSCM_GWH

        color = PASTEL_COLORS[idx % len(PASTEL_COLORS)]

        plt.plot(
            corr,
            vals,
            '-',
            label=f"Subsidy {int(subsidy)} €/MWh",
            color=color,
        )

        plt.fill_between(
            corr,
            vals - se_vals,
            vals + se_vals,
            color=color,
            alpha=0.2,
        )

    plt.xlabel("Correlation")
    plt.ylabel("Average H2 Production (GWh)")

    shock_label = "Long-term" if shock_type == "lt" else "Day Ahead"
    if vol_multiplier is None:
        plt.title(
            f"Average Hydrogen Production by Correlation ({shock_label} Price Shock)\n(Averaged over Volatility Multipliers)"
        )
        filename = f"avg_h2_production_correlation_shock_{shock_type}_avg_vol.png"
    else:
        plt.title(
            f"Average Hydrogen Production by Correlation ({shock_label} Price Shock)\n(Volatility Multiplier = {vol_multiplier})"
        )
        filename = f"avg_h2_production_correlation_shock_{shock_type}_vol_{vol_multiplier}.png"

    plt.grid(alpha=0.3, axis="y")
    plt.legend(loc="lower left", fontsize="small")

    plt.savefig(os.path.join(folder, filename), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Path to your result files
    folder_path = "study_case_model/figures/volatility_stress/07052026"   # <-- change this
    file_pattern = os.path.join(folder_path, "stress_test_results_*")

    combined_df = collect_and_combine_results(folder_path, file_pattern)

    plot_cvar(combined_df, shock_type = "lt")
    plot_cvar(combined_df, shock_type = "st")

    plot_objective(combined_df, shock_type = "lt")
    plot_objective(combined_df, shock_type = "st")

    # combined_df = add_stage_capacity_fractions(combined_df, folder_path)
    # plot_stage_capacity_fractions_combined(combined_df)

    # plot_stage_capacities_fraction_correlation(combined_df, shock_type="lt", capacity_type="entry", stage = 2)
    # plot_stage_capacities_fraction_correlation(combined_df, shock_type="lt", capacity_type="exit", stage = 2)
    # plot_stage_capacities_fraction_correlation(combined_df, shock_type="st", capacity_type="entry", stage = 3)
    # plot_stage_capacities_fraction_correlation(combined_df, shock_type="st", capacity_type="exit", stage = 3)

    plot_avg_hydrogen_production(combined_df, shock_type="lt")
    plot_avg_hydrogen_production(combined_df, shock_type="st")
    plot_avg_hydrogen_production_vs_correlation(combined_df, shock_type="lt")
    plot_avg_hydrogen_production_vs_correlation(combined_df, shock_type="st")
    plot_avg_hydrogen_production_vs_correlation(combined_df, shock_type="lt", vol_multiplier=4.0)
    plot_avg_hydrogen_production_vs_correlation(combined_df, shock_type="lt", vol_multiplier=8.0)

