import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os


# ---------------------------
# LOAD DATA
# ---------------------------
def load_series_data(files, indices, sample_sizes):
    dfs = []
    for file, n in zip(files, sample_sizes):
        df = pd.read_csv(file)

        # create index 1..N
        df["index"] = indices

        df["n"] = n
        dfs.append(df)

    return dfs


# ---------------------------
# COMBINE FUNCTION
# ---------------------------
def combine_group_series(group):
    N = group["n"].sum()

    result = {}
    result["index"] = group["index"].iloc[0]

    ns = group["n"]

    # ---- TIME ----
    means = group["time_mean"]
    sds = group["time_std"]  # assumed std

    weighted_mean = np.sum(ns * means) / N

    pooled_var = (
        np.sum((ns - 1) * (sds ** 2) + ns * (means - weighted_mean) ** 2)
    ) / (N - 1)

    pooled_sd = np.sqrt(pooled_var)
    pooled_se = pooled_sd / np.sqrt(N)

    result["time_mean"] = weighted_mean
    result["time_stderr"] = pooled_se

    # ---- OBJECTIVE ----
    means = group["objective_mean"]
    sds = group["objective_std"]

    weighted_mean = np.sum(ns * means) / N

    pooled_var = (
        np.sum((ns - 1) * (sds ** 2) + ns * (means - weighted_mean) ** 2)
    ) / (N - 1)

    pooled_sd = np.sqrt(pooled_var)
    pooled_se = pooled_sd / np.sqrt(N)

    result["objective_mean"] = weighted_mean
    result["objective_stderr"] = pooled_se

    return pd.Series(result)


# ---------------------------
# PLOT FUNCTION
# ---------------------------
def plot_results(csv_path, folder="figures/"):
    os.makedirs(folder, exist_ok=True)

    df = pd.read_csv(csv_path)

    x = df["index"].values

    # ---------------------------
    # TIME PLOT
    # ---------------------------
    plt.figure(figsize=(10, 6))

    plt.plot(x, df["time_mean"], marker='o', linestyle='-', color="#FF8692")

    plt.errorbar(
        x,
        df["time_mean"],
        yerr=df["time_stderr"],
        fmt='none',
        ecolor="#FF8692",
        capsize=3
    )

    plt.xlabel("Amount of homogeneous splits")
    plt.ylabel("Solve Time (seconds)")
    plt.title("Solve Time for different amount of homogeneous splits")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(folder, "time_series.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved time plot to {save_path}")

    # ---------------------------
    # OBJECTIVE PLOT
    # ---------------------------
    plt.figure(figsize=(10, 6))

    plt.plot(x, df["objective_mean"], marker='o', linestyle='-', color="#FF8692")

    plt.errorbar(
        x,
        df["objective_mean"],
        yerr=df["objective_stderr"],
        fmt='none',
        ecolor="#FF8692",
        capsize=3
    )

    plt.xlabel("Amount of homogeneous splits")
    plt.ylabel("Objective Value")
    plt.title("Objective for different amount of homogeneous splits")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(folder, "objective_series.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved objective plot to {save_path}")


# ---------------------------
# MAIN
# ---------------------------
def main():
    files = [
        "study_case_model/figures/examine_speed/run23326/splits_solve_times.csv",
        "study_case_model/figures/examine_speed/solve_times/splits_solve_times.csv",
    ]
    sample_sizes = [10, 10]

    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", default=files)
    parser.add_argument("--ns", nargs="+", type=int, default=sample_sizes)
    parser.add_argument("--output", default="study_case_model/figures/examine_speed/combined_runs/combined_homogeneous.csv")

    args = parser.parse_args()

    if len(args.files) != len(args.ns):
        raise ValueError("Mismatch between files and sample sizes")

    dfs = load_series_data(args.files, [6,11,16,21,26], args.ns)

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

    folder = "study_case_model/figures/examine_speed/combined_runs/homogeneous_splits/"
    plot_results(args.output, folder)


if __name__ == "__main__":
    main()