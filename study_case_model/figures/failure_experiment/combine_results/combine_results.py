from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt

EXPERIMENT = "run_13426_2"
LOAD_ONE_RUN = None  # "run0"
MINIMUM_RUNS = 2
HYDROGEN_MSCM_MWH = 2.78 * 1000 

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

# ---------------------------
# LOAD SNAPSHOT
# ---------------------------
def load_snapshot(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

# ---------------------------
# COMPUTE METRICS
# ---------------------------
def compute_h2(snapshot):
    try:
        h2_dict = snapshot["expressions"]["h2_production"]
        values = list(h2_dict.values())
        return np.mean(values), np.std(values)
    except KeyError:
        return None, None

def compute_objective(snapshot):
    try:
        return list(snapshot["objectives"].values())[0]
    except:
        return None

# ---------------------------
# FAILURE CASE PROCESSING
# ---------------------------
def process_failed_sub(folder):
    results = []

    for run_dir in sorted(Path(folder).glob("run*/")):
        files = list(run_dir.glob("*.pkl"))

        if LOAD_ONE_RUN and run_dir.name != LOAD_ONE_RUN:
            continue

        if not files:
            continue

        snapshot = load_snapshot(files[0])
        val, std = compute_h2(snapshot)

        if val is not None:
            results.append(val)

    if len(results) == 0:
        return None, None, 0, []

    results = np.array(results)
    mean = np.mean(results)
    stderr = np.std(results, ddof=1) / np.sqrt(len(results))

    return mean, stderr, len(results), results

def analyze_failed_experiment(base_folder):
    summary = {}
    base = Path(base_folder)

    print("H2 Production by FAILED component:")

    for failed_dir in sorted(base.glob("maxh2_*/")):
        FAILED = failed_dir.name.replace("maxh2_", "")

        for sub_dir in sorted(failed_dir.glob("sub*/")):
            SUBSIDY = float(sub_dir.name.replace("sub", ""))

            mean, stderr, n, results = process_failed_sub(sub_dir)

            if n >= MINIMUM_RUNS:
                summary.setdefault(FAILED, {
                    "subs": [],
                    "mean": [],
                    "se": []
                })

                summary[FAILED]["subs"].append(SUBSIDY)
                summary[FAILED]["mean"].append(mean)
                summary[FAILED]["se"].append(stderr)

                print(f"FAILED={FAILED}, sub={SUBSIDY} | mean={mean:.4f}, se={stderr:.4f}, n={n}")

    return summary

# ---------------------------
# BASELINE (NO FAILURE)
# ---------------------------
def process_subsidy_folder(folder):
    results = []

    for run_dir in sorted(Path(folder).glob("run*/")):
        files = list(run_dir.glob("*.pkl"))

        if LOAD_ONE_RUN and run_dir.name != LOAD_ONE_RUN:
            continue

        if not files:
            continue

        snapshot = load_snapshot(files[0])
        val, std = compute_h2(snapshot)

        if val is not None:
            results.append(val)

    if len(results) == 0:
        return None, None, 0, []

    results = np.array(results)
    mean = np.mean(results)
    stderr = np.std(results, ddof=1) / np.sqrt(len(results))

    return mean, stderr, len(results), results

def analyze_subsidy_experiment(base_folder, deviation = 0.0, sub_values = [30.0, 45.0, 70.0]):
    summary = {}
    base = Path(base_folder)

    print("\nBaseline H2 Production (no failures):")

    for sub in sub_values:
        sub_dir = base / f"dev{deviation}" / f"sub{sub}"
        print(f"Looking for folder: {sub_dir}")
        if not sub_dir.exists():
            print(f"Missing folder for subsidy {sub}")
            continue

        print(f"Processing subsidy {sub}...")
        mean, stderr, n, results = process_subsidy_folder(sub_dir)

        if n >= MINIMUM_RUNS:
            summary[sub] = {
                "mean": mean,
                "se": stderr,
                "n": n
            }

            print(f"sub={sub} | mean={mean:.4f}, se={stderr:.4f}, n={n}")

    return summary

# ---------------------------
# PLOTTING
# ---------------------------
def plot_remaining_h2(summary, baseline_summary=None, output_path=None):
    failed_labels = []
    all_means = []
    all_errors = []
    subsidy_values = []

    # Collect subsidies
    for data in summary.values():
        subsidy_values.extend(data["subs"])
    subsidy_values = sorted(set(subsidy_values))

    # Collect data
    for FAILED, data in summary.items():
        failed_labels.append(FAILED)
        all_means.append(data["mean"])
        all_errors.append(data["se"])

    x = np.arange(len(failed_labels))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for plot_idx, subsidy in enumerate(subsidy_values[:3]):
        ax = axes[plot_idx]

        means_for_subsidy = [
            means[plot_idx] if plot_idx < len(means) else 0
            for means in all_means
        ]
        errors_for_subsidy = [
            errors[plot_idx] if plot_idx < len(errors) else 0
            for errors in all_errors
        ]

        ax.bar(
            x,
            means_for_subsidy,
            width,
            yerr=errors_for_subsidy,
            capsize=5,
            color=PASTEL_COLORS[plot_idx],
            label=f"Remaining H2 (Sub {subsidy})"
        )

        # ---- BASELINE LINE ----
        if baseline_summary and subsidy in baseline_summary:
            baseline = baseline_summary[subsidy]["mean"]
            ax.axhline(
                baseline,
                linestyle="--",
                linewidth=2,
                color="black",
                label="Baseline (no failure)"
            )

        ax.set_xticks(x)
        ax.set_xticklabels(failed_labels, rotation=45, ha="right")
        ax.set_ylabel("Hydrogen Production")
        ax.set_xlabel("Failed Pipeline / Generation Node")
        ax.set_title(f"Impact of Failures (Subsidy {subsidy})")
        ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()

    plt.close()

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    base_folder_failure = f"study_case_model/figures/failure_experiment/{EXPERIMENT}/"
    base_folder_no_failure = f"study_case_model/figures/subsidy_experiment/combined_runs_new/"

    sub_values = [30.0, 45.0, 70.0]

    # Failure results
    summary_failure = analyze_failed_experiment(base_folder_failure)

    # Baseline results
    baseline_summary = analyze_subsidy_experiment(base_folder_no_failure, deviation=0.0, sub_values=sub_values)

    print("\nBaseline summary:")
    for sub, data in baseline_summary.items():
        print(sub, data)

    # Plot
    plot_remaining_h2(
        summary_failure,
        baseline_summary=baseline_summary,
        output_path="study_case_model/figures/failure_experiment/combine_results/remaining_h2.png"
    )