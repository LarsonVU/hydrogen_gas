import pandas as pd
import numpy as np
import argparse
from functools import reduce
import matplotlib.pyplot as plt
from pathlib import Path
import os
import matplotlib.colors as mcolors


def load_data(files, sample_sizes):
    dfs = []
    for file, n in zip(files, sample_sizes):
        df = pd.read_csv(file)
        df["n"] = n
        dfs.append(df)
    return dfs

def combine_group(group):
    # total sample size
    N = group["n"].sum()

    result = {}

    # keep identifiers
    for col in ["m_2", "m_3"]:
        result[col] = group[col].iloc[0]

    means = group["mean"]
    ses = group["stderr"]   # this is already standard error
    ns = group["n"]

    # convert SE -> SD
    sds = ses * np.sqrt(ns)

    # pooled mean
    weighted_mean = np.sum(ns * means) / N

    # pooled variance (accounts for within + between variance)
    pooled_var = (
        np.sum((ns - 1) * (sds ** 2) + ns * (means - weighted_mean) ** 2)
    ) / (N - 1)

    pooled_sd = np.sqrt(pooled_var)

    # convert back to SE
    pooled_se = pooled_sd / np.sqrt(N)

    result["mean"] = weighted_mean
    result["stderr"] = pooled_se

    return pd.Series(result)

def load_and_plot_solve_time_matrix(csv_path, folder="figures/"):
    os.makedirs(folder, exist_ok=True)

    df = pd.read_csv(csv_path)

    # unique sorted values
    m_2_values = sorted(df["m_2"].unique())
    m_3_values = sorted(df["m_3"].unique())

    # build lookup dict: (m_2, m_3) -> (mean, stderr)
    solve_times = {
        (row["m_2"], row["m_3"]): (row["mean"], row["stderr"])
        for _, row in df.iterrows()
    }

    # build matrix Z (means)
    Z = np.array([
        [solve_times.get((m_2, m_3), (np.nan, np.nan))[0] for m_3 in m_3_values]
        for m_2 in m_2_values
    ])

    # Define your pastel colors (example palette)
    pastel_colors = ["#82C9FF", "#7AFF97",]

    # Create a continuous colormap
    continuous_pastel = mcolors.LinearSegmentedColormap.from_list(
        "continuous_pastel", pastel_colors
        )


    plt.figure(figsize=(10, 6))
    im = plt.imshow(Z, origin='lower', aspect='auto', cmap=continuous_pastel)
    cbar = plt.colorbar(im, label='Mean Solve Time (seconds)')

    # --- add borders ---
    ax = plt.gca()


    # set minor ticks at cell boundaries
    ax.set_xticks(np.arange(-0.5, len(m_3_values), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(m_2_values), 1), minor=True)

    # draw gridlines
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)

    # remove minor tick marks
    ax.tick_params(which='minor', bottom=False, left=False)

    # annotate with mean ± stderr
    for i, m_2 in enumerate(m_2_values):
        for j, m_3 in enumerate(m_3_values):
            mean, stderr = solve_times.get((m_2, m_3), (np.nan, np.nan))

            if not np.isnan(mean):
                plt.text(
                    j, i,
                    f'{mean:.2f}±{stderr:.2f}',
                    ha='center', va='center',
                    color='black',
                    fontsize=9
                )

    plt.xticks(ticks=range(len(m_3_values)), labels=m_3_values)
    plt.yticks(ticks=range(len(m_2_values)), labels=m_2_values)

    plt.xlabel('Branches in Stage 3')
    plt.ylabel('Branches in Stage 2')
    plt.title('Solve Time Matrix')

    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'solve_time_matrix_branches.png'))
    plt.show()


def main():
    files = [
        "study_case_model/figures/examine_speed/run23326/branch_solve_times.csv",
        "study_case_model/figures/examine_speed/solve_times/branch_solve_times.csv",
    ]
    sample_number = [10,10]

    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", default =files, help="CSV files")
    parser.add_argument("--ns", nargs="+", type=int, default=sample_number, help="Sample sizes per file")
    parser.add_argument("--output", default="study_case_model/figures/examine_speed/combined_runs/combined_branches.csv")

    args = parser.parse_args()

    if len(args.files) != len(args.ns):
        raise ValueError("Number of files and sample sizes must match")

    dfs = load_data(args.files, args.ns)

    # concatenate all data
    full_df = pd.concat(dfs, ignore_index=True)

    # group and combine
    combined = (
        full_df
        .groupby(["m_2", "m_3"], as_index=False)
        .apply(combine_group)
        .reset_index(drop=True)
    )

    combined.to_csv(args.output, index=False)
    print(f"Saved combined results to {args.output}")

    folder = "study_case_model/figures/examine_speed/combined_runs/"
    load_and_plot_solve_time_matrix(args.output, folder)


if __name__ == "__main__":
    main()
    