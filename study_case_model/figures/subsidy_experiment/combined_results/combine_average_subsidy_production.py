import os
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os


# ---------------------------
# LOAD SNAPSHOT
# ---------------------------
def load_snapshot(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


# ---------------------------
# COMPUTE METRIC (EDIT THIS)
# ---------------------------
def compute_h2(snapshot):
    """
    Example: compute mean H2 production from saved variables.
    Adjust key names if needed.
    """
    try:
        h2_dict = snapshot["expressions"]["h2_production"]
        values = list(h2_dict.values())
        return np.mean(values), np.std(values)
    except KeyError:
        return None

def compute_objective(snapshot):
    """
    Extract objective value from snapshot.
    Assumes single objective.
    """
    try:
        return list(snapshot["objectives"].values())[0]
    except:
        return None

# ---------------------------
# PROCESS SINGLE (dev, sub)
# ---------------------------
def process_dev_sub(folder):
    """
    Loop over all runs inside a (dev, sub) folder.
    """
    results = []

    for run_dir in sorted(Path(folder).glob("run*/")):
        files = list(run_dir.glob("*.pkl"))

        if not files:
            continue

        snapshot = load_snapshot(files[0])
        val, std = compute_h2(snapshot)
        if val is not None:
            results.append(val)


    if len(results) == 0:
        return None, None, 0

    results = np.array(results)

    mean = np.mean(results)
    stderr = np.std(results, ddof =1) /len(results)

    return mean, stderr, len(results), results


# ---------------------------
# MAIN LOOP
# ---------------------------
def analyze_experiment(base_folder):
    summary = {}

    base = Path(base_folder)

    for dev_dir in sorted(base.glob("dev*/")):
        dev_value = float(dev_dir.name.replace("dev", ""))

        for sub_dir in sorted(dev_dir.glob("sub*/")):
            sub_value = float(sub_dir.name.replace("sub", ""))

            mean, stderr, n, results = process_dev_sub(sub_dir)

            if n > 0:
                summary.setdefault(dev_value, {
                                    "subsidy": [],
                                    "mean": [],
                                    "se": []
                                })

                summary[dev_value]["subsidy"].append(sub_value)
                summary[dev_value]["mean"].append(mean)
                summary[dev_value]["se"].append(stderr)

                print(f"dev={dev_value}, sub={sub_value} | mean={mean:.4f}, stderr={stderr:.4f}, n={n}")
    return summary


def plot_hydrogen_production(h2_dict, folder):
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(10, 5))

    for label, stats in sorted(h2_dict.items()):
        # sort by subsidy to ensure clean lines
        sorted_data = sorted(
            zip(stats["subsidy"], stats["mean"], stats["se"]),
            key=lambda x: x[0]
        )

        subs, means, ses = zip(*sorted_data)

        plt.errorbar(
            subs,
            means,
            yerr=ses,
            fmt='o-',
            capsize=5,
            label=f"Deviation {label}"
        )

    plt.xlabel('Subsidy (Euro/MWh)')
    plt.ylabel('Hydrogen Production')
    plt.title('Hydrogen Production vs Subsidy')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(folder, "hydrogen_production_vs_subsidy.png"))
    plt.close()

def analyze_objectives(base_folder):
    summary = {}

    base = Path(base_folder)

    for dev_dir in sorted(base.glob("dev*/")):
        dev_value = float(dev_dir.name.replace("dev", ""))

        for sub_dir in sorted(dev_dir.glob("sub*/")):
            sub_value = float(sub_dir.name.replace("sub", ""))

            results = []

            for run_dir in sorted(sub_dir.glob("run*/")):
                files = list(run_dir.glob("*.pkl"))
                if not files:
                    continue

                with open(files[0], "rb") as f:
                    snapshot = pickle.load(f)

                val = compute_objective(snapshot)
                if val is not None:
                    results.append(val)

            if len(results) == 0:
                continue

            results = np.array(results)

            mean = np.mean(results)
            se = np.std(results, ddof=1) / np.sqrt(len(results))

            # store in structured format
            summary.setdefault(dev_value, {
                "subsidy": [],
                "mean": [],
                "se": []
            })

            summary[dev_value]["subsidy"].append(sub_value)
            summary[dev_value]["mean"].append(mean)
            summary[dev_value]["se"].append(se)

    return summary

def plot_objective_values(objective_dict, folder):
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(10, 5))

    for label, stats in sorted(objective_dict.items()):
        # ensure correct ordering by subsidy
        sorted_data = sorted(
            zip(stats["subsidy"], stats["mean"], stats["se"]),
            key=lambda x: x[0]
        )

        subs, means, ses = zip(*sorted_data)

        plt.errorbar(
            subs,
            means,
            yerr=ses,
            fmt='o-',
            capsize=5,
            label=f"Deviation {label}"
        )

    plt.xlabel('Subsidy (Euro/MWh)')
    plt.ylabel('Objective Value')
    plt.title('Objective Value vs Subsidy')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(folder, "objective_vs_subsidy.png"))
    plt.close()

# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    base_folder = "study_case_model/figures/subsidy_experiment/run_24326"

    results = analyze_experiment(base_folder)

    plot_hydrogen_production(results, folder="study_case_model/figures/subsidy_experiment/combined_results")

    objective_dict = analyze_objectives("study_case_model/figures/subsidy_experiment/run_24326")

    plot_objective_values(objective_dict, folder="study_case_model/figures/subsidy_experiment/combined_results")