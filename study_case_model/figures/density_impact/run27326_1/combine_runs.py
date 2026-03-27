import pandas as pd
import numpy as np
import argparse
from functools import reduce
import matplotlib.pyplot as plt
from pathlib import Path


MSCM_to_GWH = 27.8

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
    for col in ["subsidy", "density"]:
        result[col] = group[col].iloc[0]

    # --- process each metric ---
    metrics = ["h2", "pressure"]

    for m in metrics:
        means = group[f"{m}_mean"]
        ses = group[f"{m}_std"]  # assumed SE
        ns = group["n"]

        # convert SE -> SD
        sds = ses # * np.sqrt(ns)

        # pooled mean
        weighted_mean = np.sum(ns * means) / N

        # pooled variance (with mean correction)
        pooled_var = (
            np.sum((ns - 1) * (sds ** 2) + ns * (means - weighted_mean) ** 2)
        ) / (N - 1)

        pooled_sd = np.sqrt(pooled_var)

        # convert back to SE
        pooled_se = pooled_sd / np.sqrt(N)

        result[f"{m}_mean"] = weighted_mean
        result[f"{m}_std"] = pooled_se

    return pd.Series(result)

def load_and_plot(csv_path, folder="figures/", runs=None):
    """
    Load results from CSV and generate H2 and Pressure plots.

    Parameters:
    - csv_path: path to input CSV
    - folder: output folder for plots
    - runs: if provided, assumes std column is SD and converts to SE via /sqrt(runs)
            if None, assumes std column is already SE
    """
    # Soft pastel palette
    pastel_colors = [
        "#FF8692",  # pastel red
        "#7AFF97",  # pastel green
        "#82C9FF",  # pastel blue
        "#DB97E3",  # pastel purple
        "#FFFF82",  # pastel yellow
    ]


    df = pd.read_csv(csv_path)

    # unique sorted values
    subsidies = sorted(df["subsidy"].unique())
    densities = sorted(df["density"].unique())

    # build results dict
    results = {
        s: {
            d: {
                "h2_mean": None,
                "h2_std": None,
                "pressure_mean": None,
                "pressure_std": None,
            }
            for d in densities
        }
        for s in subsidies
    }

    for _, row in df.iterrows():
        s = row["subsidy"]
        d = row["density"]

        results[s][d] = {
            "h2_mean": row["h2_mean"],
            "h2_std": row["h2_std"],
            "pressure_mean": row["pressure_mean"],
            "pressure_std": row["pressure_std"],
        }

    Path(folder).mkdir(parents=True, exist_ok=True)

    # helper: determine error
    def get_err(std):
        if runs is None:
            return std  # already SE
        return std / np.sqrt(runs)  # convert SD -> SE

    # ---- H2 ----
    plt.figure(figsize=(10, 6))

    for i,s in enumerate(subsidies):
        means = [results[s][d]["h2_mean"] *MSCM_to_GWH for d in densities]
        errs = [get_err(results[s][d]["h2_std"]) *MSCM_to_GWH for d in densities]

        plt.errorbar(densities, means, yerr=errs, marker="o", capsize=5, label=f"Subsidy {s}",
        color=pastel_colors[i % len(pastel_colors)])

    plt.xlabel("Density Bounds")
    plt.ylabel("Hydrogen Production (Gwh)")
    plt.title("H2 Production vs Density Bounds")
    plt.grid()
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(folder + "h2_vs_density.png")
    plt.show()

    # ---- Pressure ----
    plt.figure(figsize=(10, 6))

    for i,s in enumerate(subsidies):
        means = [results[s][d]["pressure_mean"] for d in densities]
        errs = [get_err(results[s][d]["pressure_std"]) for d in densities]

        plt.errorbar(densities, means, yerr=errs, marker="o", capsize=5, label=f"Subsidy {s}", color=pastel_colors[i % len(pastel_colors)])

    plt.xlabel("Density Bounds")
    plt.ylabel("Pressure Cost (Euro)")
    plt.title("Pressure Cost vs Density Bounds")
    plt.grid()
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(folder + "pressure_vs_density.png")
    plt.show()

    print(f"Plots saved in {folder}")

def main():
    files = [
        "study_case_model/figures/density_impact/run27326_1/density_impact.csv",
    ]
    sample_number = [20]

    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", default =files, help="CSV files")
    parser.add_argument("--ns", nargs="+", type=int, default=sample_number, help="Sample sizes per file")
    parser.add_argument("--output", default="study_case_model/figures/density_impact/run27326_1/combined.csv")

    args = parser.parse_args()

    if len(args.files) != len(args.ns):
        raise ValueError("Number of files and sample sizes must match")

    dfs = load_data(args.files, args.ns)

    # concatenate all data
    full_df = pd.concat(dfs, ignore_index=True)

    # group and combine
    combined = (
        full_df
        .groupby(["subsidy", "density"], as_index=False)
        .apply(combine_group)
        .reset_index(drop=True)
    )

    combined.to_csv(args.output, index=False)
    print(f"Saved combined results to {args.output}")

    folder = "study_case_model/figures/density_impact/run27326_1/"
    load_and_plot(args.output, folder, sum(sample_number))


if __name__ == "__main__":
    main()
    