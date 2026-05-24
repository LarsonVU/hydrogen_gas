import pandas as pd
from glob import glob
import seaborn as sns
# Create heatmaps
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from itertools import combinations
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


# Read all CSV files matching the pattern
EXPERIMENT_ID = 14526
csv_files = glob(f'study_case_model/figures/vss_experiment/{EXPERIMENT_ID}/vss_results_*.csv')

# Read and combine all files into one dataframe
dfs = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(dfs, ignore_index=True)

# Print the head
print(combined_df.describe()["VSS"])
# Group by subsidy, deviation, branches_s2, branches_s3 and calculate stats
grouped = combined_df.groupby(['subsidy', 'deviation', 'branches_s2', 'branches_s3']).agg({
    'VSS': ['mean', 'std'],
    'EVPI': ['mean', 'std'],
    'EVPI_cond' : ['mean', 'std'],
    'RP': 'mean',
    'run': 'count'  # Count the number of samples in each group
})

# Calculate VSS and EVPI as percentage of RP
grouped[('VSS_pct', 'mean')] = (grouped[('VSS', 'mean')] / grouped[('RP', 'mean')] * 100)
grouped[('VSS_pct', 'std')] = (grouped[('VSS', 'std')] / grouped[('RP', 'mean')] * 100)
grouped[('EVPI_pct', 'mean')] = (grouped[('EVPI', 'mean')] / grouped[('RP', 'mean')] * 100)
grouped[('EVPI_pct', 'std')] = (grouped[('EVPI', 'std')] / grouped[('RP', 'mean')] * 100)
grouped[('EVPI_cond_pct', 'mean')] = (grouped[('EVPI_cond', 'mean')] / grouped[('RP', 'mean')] * 100)
grouped[('EVPI_cond_pct', 'std')] = (grouped[('EVPI_cond', 'std')] / grouped[('RP', 'mean')] * 100)
grouped[('VSS_pct', 'se')] = grouped[('VSS_pct', 'std')] / np.sqrt(grouped[('run', 'count')])
grouped[('EVPI_pct', 'se')] = grouped[('EVPI_pct', 'std')] / np.sqrt(grouped[('run', 'count')])
grouped[('EVPI_cond_pct', 'se')] = grouped[('EVPI_cond_pct', 'std')] / np.sqrt(grouped[('run', 'count')])



for deviation_val in grouped.index.get_level_values('deviation').unique():
    subset = grouped.loc[(slice(None), deviation_val, slice(None), slice(None)), :]

    # Means + SE
    vss_mean = subset[('VSS_pct', 'mean')].unstack(['branches_s2', 'branches_s3']).T
    vss_se   = subset[('VSS_pct', 'se')].unstack(['branches_s2', 'branches_s3']).T

    evpi_mean = subset[('EVPI_pct', 'mean')].unstack(['branches_s2', 'branches_s3']).T
    evpi_se   = subset[('EVPI_pct', 'se')].unstack(['branches_s2', 'branches_s3']).T

    evpi_cond_mean = subset[('EVPI_cond_pct', 'mean')].unstack(['branches_s2', 'branches_s3']).T
    evpi_cond_se = subset[('EVPI_cond_pct', 'se')].unstack(['branches_s2', 'branches_s3']).T

    print(f"\n--- Deviation: {int(deviation_val * 100)}% ---")
    print(f"EVPI means: \n{evpi_mean}")
    print(f"CEVPI means: \n{evpi_cond_mean}")

    # Keep structure for grouping
    original_index = vss_mean.index

    # --- Annotation function ---
    def format_annot(mean_df, se_df):
        annot = mean_df.copy().astype(str)
        for i in range(mean_df.shape[0]):
            for j in range(mean_df.shape[1]):
                annot.iloc[i, j] = f"{mean_df.iloc[i,j]:.2f}\n±{se_df.iloc[i,j]:.2f}"
        return annot

    vss_annot = format_annot(vss_mean, vss_se)
    evpi_annot = format_annot(evpi_mean, evpi_se)
    evpi_cond_annot = format_annot(evpi_cond_mean, evpi_cond_se)

    # --- Figure ---
    if deviation_val not in [0.0, 1.0]:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    else: 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Colormap
    pastel_colors = ["#82C9FF", "#7AFF97"]
    continuous_pastel = mcolors.LinearSegmentedColormap.from_list(
        "continuous_pastel", pastel_colors
    )

    cbars = []
    # --- Heatmaps ---
    hm1 = sns.heatmap(
        vss_mean,
        annot=vss_annot,
        fmt='',
        ax=ax1,
        cmap=continuous_pastel,
        vmin=0.35,
        vmax=0.45,
        cbar_kws={'label': 'VSS (%)'}
    )

    # --- Colorbar formatting (ticks + borders) ---
    cbar1 = hm1.collections[0].colorbar
    cbar1.set_ticks([0.35, 0.375, 0.40, 0.425, 0.45])
    cbars.append(cbar1)

    if deviation_val in [0.0, 1.0]:
        hm2 = sns.heatmap(
        evpi_mean,
        annot=evpi_annot,
        fmt='',
        ax=ax2,
        cmap=continuous_pastel,
        vmin=0.05,
        vmax=0.10,
        cbar_kws={'label': 'EVPI (%)'}
    )
        cbar2 = hm2.collections[0].colorbar
        cbar2.set_ticks([0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
        cbars.append(cbar2)
    else:
        hm2 = sns.heatmap(
        evpi_mean,
        annot=evpi_annot,
        fmt='',
        ax=ax2,
        cmap=continuous_pastel,
        vmin=0.0,
        vmax=1.5,
        cbar_kws={'label': 'EVPI (%)'}
        )
        
        hm3 = sns.heatmap(
            evpi_cond_mean,
            annot=evpi_cond_annot,
            fmt='',
            ax=ax3,
            cmap=continuous_pastel,
            vmin=0.05,
            vmax=0.10,
            cbar_kws={'label': 'Conditional EVPI (%)'}
        )
        cbar2 = hm2.collections[0].colorbar
        cbar2.set_ticks([0.0, 0.50, 1.0, 1.5])
        cbars.append(cbar2)

        cbar3 = hm3.collections[0].colorbar
        cbar3.set_ticks([0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
        cbars.append(cbar3)
    
    for cbar in cbars:
        cbar.ax.tick_params(length=6, width=1.2, direction='out')

    # --- S3 labels only ---
    s3_labels = [f"S3={s3}" for s2, s3 in original_index]
    ax1.set_yticklabels(s3_labels, rotation=90)
    ax2.set_yticklabels(s3_labels, rotation=90)

    x_labels = [f"{int(x[0])} Euro/MWh" for x in vss_mean.columns]
    ax1.set_xticklabels(x_labels, rotation=0)
    ax2.set_xticklabels(x_labels, rotation=0)

    # --- S2 separators ---
    def add_s2_separators(ax, index):
        last_s2 = None
        for i, (s2, s3) in enumerate(index):
            if last_s2 is not None and s2 != last_s2:
                ax.axhline(i, color='black', linewidth=2)
            last_s2 = s2

    add_s2_separators(ax1, original_index)
    add_s2_separators(ax2, original_index)

    # --- S2 group labels (vertical, centered in axis coords) ---
    def add_s2_group_labels(ax, index):
        s2_vals = [s2 for s2, _ in index]

        start = 0
        n = len(index)

        for i in range(1, n + 1):
            if i == n or s2_vals[i] != s2_vals[start]:
                mid = (start + i - 1) / 2
                y_pos = 1 - (mid + 0.5) / n

                ax.text(
                    -0.08,
                    y_pos,
                    f"S2={s2_vals[start]}",
                    va='center',
                    ha='center',
                    rotation=90,
                    fontsize=10,
                    fontweight='bold',
                    transform=ax.transAxes
                )
                start = i

    add_s2_group_labels(ax1, original_index)
    add_s2_group_labels(ax2, original_index)

    # --- Titles & labels ---
    ax1.set_title(f'VSS (deviation={int(deviation_val * 100)}%)')
    ax2.set_title(f'EVPI (deviation={int(deviation_val * 100)}%)')

    ax1.set_ylabel('Branches', labelpad=40)
    ax2.set_ylabel('Branches', labelpad=40)

    ax1.set_xlabel('Subsidy')
    ax2.set_xlabel('Subsidy')

    if deviation_val not in [0.0, 1.0]:
        ax3.set_yticklabels(s3_labels, rotation=90)
        ax3.set_xticklabels(x_labels, rotation=0)

        add_s2_separators(ax3, original_index)
        add_s2_group_labels(ax3, original_index)

        ax3.set_title(f'Conditional EVPI (deviation={int(deviation_val * 100)}%)')
        ax3.set_ylabel('Branches', labelpad=40)
        ax3.set_xlabel('Subsidy')

    plt.tight_layout()

    plt.savefig(
        f'study_case_model/figures/vss_experiment/{EXPERIMENT_ID}/vss_evpi_deviation_{deviation_val}.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()



combined_df['VSS_pct'] = (combined_df['VSS'] / combined_df['RP']) * 100
combined_df['EVPI_pct'] = (combined_df['EVPI'] / combined_df['RP']) * 100

# Statistical testing across all subsidies
print(f"\n--- Deviation: 0% and 100% combined tests ---")


# Variables to test
group_columns = ['branches_s2', 'branches_s3', 'subsidy']
metrics = ['VSS_pct', 'EVPI_pct']

for group_col in group_columns:
    print(f"\n{'='*60}")
    print(f"GROUP VARIABLE: {group_col}")
    print(f"{'='*60}")

    # Unique groups
    groups = sorted(combined_df[group_col].dropna().unique())

    for metric in metrics:
        print(f"\n--- {metric} ---")

        results = []

        # Pairwise group comparisons
        for g1, g2 in combinations(groups, 2):
            sample1 = combined_df[
                (combined_df[group_col] == g1) &
                (combined_df['deviation'].isin([0.0, 1.0]))
            ][metric].values

            sample2 = combined_df[
                (combined_df[group_col] == g2) &
                (combined_df['deviation'].isin([0.0, 1.0]))
            ][metric].values

            if len(sample1) > 0 and len(sample2) > 0:

                # Means
                mean1 = sample1.mean()
                mean2 = sample2.mean()

                stat, p_value = mannwhitneyu(
                    sample1,
                    sample2,
                    alternative='two-sided',
                    method='asymptotic',
                    use_continuity=True
                )

                results.append({
                    'comparison': f"{g1} vs {g2}",
                    'statistic': stat,
                    'p_value': p_value,
                    f'mean1': mean1,
                    f'mean2': mean2
                })

        # Multiple testing correction
        if results:
            raw_pvals = [r['p_value'] for r in results]

            reject, corrected_pvals, _, _ = multipletests(
                raw_pvals,
                method='bonferroni'
            )

            for i, r in enumerate(results):
                print(
                    f"{r['comparison']:>15} | "
                    f"mean1 = {r['mean1']:8.2f} | "
                    f"mean2 = {r['mean2']:8.2f} | "
                    f"U-stat = {r['statistic']:10.2f} | "
                    f"raw p = {r['p_value']:.4e} | "
                    f"bonf p = {corrected_pvals[i]:.4e} | "
                    f"significant = {reject[i]}"
                )