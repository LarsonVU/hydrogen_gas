import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import os


# ---------------------------
# LOAD DATA
# ---------------------------
def load_series_data(files, sample_sizes):
    dfs = []
    for file, n in zip(files, sample_sizes):
        df = pd.read_csv(file)

        # create index 1..N
        df["index"] = np.arange(1, len(df) + 1)

        df["n"] = n
        dfs.append(df)

    return dfs


# ---------------------------
# COMBINE FUNCTION
# ---------------------------
def combine_group_series(group):
    # total sample size
    N = group["n"].sum()

    result = {}

    # keep index
    result["index"] = group["index"].iloc[0]

    means = group["mean"]
    sds = group["stderr"]   # actually standard dev
    ns = group["n"]


    # pooled mean
    weighted_mean = np.sum(ns * means) / N

    # pooled variance (within + between)
    pooled_var = (
        np.sum((ns - 1) * (sds ** 2) + ns * (means - weighted_mean) ** 2)
    ) / (N - 1)

    pooled_sd = np.sqrt(pooled_var)

    # back to SE
    pooled_se = pooled_sd / np.sqrt(N)

    result["mean"] = weighted_mean
    result["stderr"] = pooled_se

    return pd.Series(result)


# ---------------------------
# PLOT FUNCTION
# ---------------------------
def load_and_plot_series(csv_path, folder="figures/"):
    os.makedirs(folder, exist_ok=True)

    df = pd.read_csv(csv_path)

    x = df["index"].values
    means = df["mean"].values
    stderrs = df["stderr"].values

    plt.figure(figsize=(10, 6))

    # plot line with dots (similar to your deviation plot)
    plt.plot(x, means, marker='o', linestyle='-', color="#FF8692")

    # optionally, add error bars as thin black lines (like little caps)
    plt.errorbar(x, means, yerr=stderrs, fmt='none', ecolor="#FF8692", capsize=3)

    plt.xlabel("Index")
    plt.ylabel("Mean Solve Time (seconds)")
    plt.title("Solve Time per Index")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(folder, "solve_time_densities.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved plot to {save_path}")

# ---------------------------
# MAIN
# ---------------------------
def main():
    files = [
        "study_case_model/figures/examine_speed/run23326/density_solve_times.csv",
        "study_case_model/figures/examine_speed/solve_times/density_solve_times.csv",
    ]
    sample_number = [10, 10]

    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", default=files, help="CSV files")
    parser.add_argument("--ns", nargs="+", type=int, default=sample_number, help="Sample sizes")
    parser.add_argument("--output", default="study_case_model/figures/examine_speed/combined_runs/combined_densities.csv")

    args = parser.parse_args()

    if len(args.files) != len(args.ns):
        raise ValueError("Number of files and sample sizes must match")

    dfs = load_series_data(args.files, args.ns)

    # combine all data
    full_df = pd.concat(dfs, ignore_index=True)

    combined = (
        full_df
        .groupby("index", as_index=False)
        .apply(combine_group_series)
        .reset_index(drop=True)
        .sort_values("index")
    )

    combined.to_csv(args.output, index=False)
    print(f"Saved combined results to {args.output}")

    # plot results
    folder = "study_case_model/figures/examine_speed/combined_runs/"
    load_and_plot_series(args.output, folder=folder)


if __name__ == "__main__":
    main()