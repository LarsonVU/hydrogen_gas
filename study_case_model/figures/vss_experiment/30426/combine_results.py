import pandas as pd
from glob import glob
import seaborn as sns
# Create heatmaps
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu


# Read all CSV files matching the pattern
csv_files = glob('study_case_model/figures/vss_experiment/30426/vss_results_*.csv')

# Read and combine all files into one dataframe
dfs = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(dfs, ignore_index=True)

# Print the head
print(combined_df.describe()["VSS"])
# Group by subsidy, deviation, branches_s2, branches_s3 and calculate stats
grouped = combined_df.groupby(['subsidy', 'deviation', 'branches_s2', 'branches_s3']).agg({
    'VSS': ['mean', 'std'],
    'EVPI': ['mean', 'std'],
    'RP': 'mean',
    'run': 'count'  # Count the number of samples in each group
})

# Calculate VSS and EVPI as percentage of RP
grouped[('VSS_pct', 'mean')] = (grouped[('VSS', 'mean')] / grouped[('RP', 'mean')] * 100)
grouped[('VSS_pct', 'std')] = (grouped[('VSS', 'std')] / grouped[('RP', 'mean')] * 100)
grouped[('EVPI_pct', 'mean')] = (grouped[('EVPI', 'mean')] / grouped[('RP', 'mean')] * 100)
grouped[('EVPI_pct', 'std')] = (grouped[('EVPI', 'std')] / grouped[('RP', 'mean')] * 100)
grouped[('VSS_pct', 'se')] = grouped[('VSS_pct', 'std')] / np.sqrt(grouped[('run', 'count')])
grouped[('EVPI_pct', 'se')] = grouped[('EVPI_pct', 'std')] / np.sqrt(grouped[('run', 'count')])



for deviation_val in grouped.index.get_level_values('deviation').unique():
    subset = grouped.loc[(slice(None), deviation_val, slice(None), slice(None)), :]

    # Means + SE
    vss_mean = subset[('VSS_pct', 'mean')].unstack(['branches_s2', 'branches_s3']).T
    vss_se   = subset[('VSS_pct', 'se')].unstack(['branches_s2', 'branches_s3']).T

    evpi_mean = subset[('EVPI_pct', 'mean')].unstack(['branches_s2', 'branches_s3']).T
    evpi_se   = subset[('EVPI_pct', 'se')].unstack(['branches_s2', 'branches_s3']).T

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

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Colormap
    pastel_colors = ["#82C9FF", "#7AFF97"]
    continuous_pastel = mcolors.LinearSegmentedColormap.from_list(
        "continuous_pastel", pastel_colors
    )

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

    # --- Colorbar formatting (ticks + borders) ---
    cbar1 = hm1.collections[0].colorbar
    cbar2 = hm2.collections[0].colorbar

    cbar1.set_ticks([0.35, 0.375, 0.40, 0.425, 0.45])
    cbar2.set_ticks([0.05, 0.06, 0.07, 0.08, 0.09, 0.10])

    for cbar in [cbar1, cbar2]:
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

    plt.tight_layout()

    plt.savefig(
        f'study_case_model/figures/vss_experiment/30426/vss_evpi_deviation_{deviation_val}.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    # Statistical testing across all subsidies
    print(f"\n--- Deviation: {int(deviation_val * 100)}% ---")

    # Split into branches_s2=2 vs others (across all subsidies)
    s2_eq_2 = combined_df[(combined_df['branches_s2'] == 2) & (combined_df['deviation'] == deviation_val)]['VSS'].values
    s2_others = combined_df[(combined_df['branches_s2'] != 2) & (combined_df['deviation'] == deviation_val)]['VSS'].values

    if len(s2_eq_2) > 0 and len(s2_others) > 0:
        # Mann-Whitney U test (non-parametric t-test)
        stat, p_value = mannwhitneyu(s2_eq_2, s2_others)
        print(f"VSS - Mann-Whitney U p-value (S2=2 vs others): {p_value:.4f}")
        
        s2_eq_2 = combined_df[(combined_df['branches_s2'] == 2) & (combined_df['deviation'] == deviation_val)]['EVPI'].values
        s2_others = combined_df[(combined_df['branches_s2'] != 2) & (combined_df['deviation'] == deviation_val)]['EVPI'].values
        
        stat, p_value = mannwhitneyu(s2_eq_2, s2_others)
        print(f"EVPI - Mann-Whitney U p-value (S2=2 vs others): {p_value:.4f}")
