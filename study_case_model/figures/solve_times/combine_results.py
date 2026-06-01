from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# COLOR PALETTE
# =========================
PASTEL_COLORS = [
    "#82C9FF",  # blue
    "#FF8692",  # red
    "#4BDA6A",  # green
    "#DB97E3",  # purple
    "#FFC085",
    "#FFFF82",  # yellow
    "#7EDCD5"
]


def combine_solve_time_csvs(base_folder):
    """
    Combine all solve_times.csv files into one dataframe
    and extract folder metadata.
    """

    base_folder = Path(base_folder)

    csv_files = list(base_folder.rglob("solve_times.csv"))

    if not csv_files:
        raise FileNotFoundError("No solve_times.csv files found.")

    dfs = []

    for file in csv_files:

        df = pd.read_csv(file)

        relative_parts = file.relative_to(base_folder).parts[:-1]

        for part in relative_parts:

            if part.startswith("sub"):
                df["subsidy"] = float(part.replace("sub", ""))

            elif part.startswith("dev"):
                df["deviation"] = float(part.replace("dev", ""))

            elif part.startswith("den"):
                df["density"] = float(part.replace("den", ""))

            elif part.startswith("split"):
                df["split"] = int(part.replace("split", ""))

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def plot_solve_times(
    df,
    x_feature,
    line_feature,
    y_feature="solve_time",
    figsize=(10, 6),
):
    """
    Plot solve times with:
    - mean line
    - shaded standard error band
    """

    plt.figure(figsize=figsize)

    unique_lines = sorted(df[line_feature].dropna().unique())

    for j, label in enumerate(unique_lines):

        subset = df[df[line_feature] == label]

        grouped = (
            subset
            .groupby(x_feature)[y_feature]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values(x_feature)
        )

        grouped["se"] = grouped["std"] / (grouped["count"] ** 0.5)

        x_vals = grouped[x_feature].values
        means = grouped["mean"].values
        ses = grouped["se"].fillna(0).values

        color = PASTEL_COLORS[j % len(PASTEL_COLORS)]

        # Line
        plt.plot(
            x_vals,
            means,
            "-",
            label=f"{line_feature}={label}",
            color=color,
        )

        # Error bands
        lower = means - ses
        upper = means + ses

        plt.fill_between(
            x_vals,
            lower,
            upper,
            color=color,
            alpha=0.2,
        )

    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title(f"{y_feature} vs {x_feature}")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()
    
base_folder = r"study_case_model/figures/solve_times"

df = combine_solve_time_csvs(base_folder)

print(df.head())

plot_solve_times(
    df,
    x_feature="homogeneous_splits",
    line_feature="deviation",
)