import os
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import folium
import geopandas as gpd
import numbers
import pandas as pd
import ast
import matplotlib.colors as mcolors

EXPERIMENT = "run_24426"
LOAD_ONE_RUN = None #"run0"
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
        if LOAD_ONE_RUN:
            if run_dir.name != LOAD_ONE_RUN:
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
    stderr = np.std(results, ddof =1) /len(results)

    return mean, stderr, len(results), results


# ---------------------------
# MAIN LOOP
# ---------------------------
def analyze_experiment(base_folder):
    summary = {}

    base = Path(base_folder)
    print("H2 Production:")
    for dev_dir in sorted(base.glob("dev*/")):
        dev_value = float(dev_dir.name.replace("dev", ""))

        for sub_dir in sorted(dev_dir.glob("sub*/")):
            sub_value = float(sub_dir.name.replace("sub", ""))
            
            mean, stderr, n, results = process_dev_sub(sub_dir)
            if n > MINIMUM_RUNS-1:
                summary.setdefault(dev_value, {
                                    "subsidy": [],
                                    "mean": [],
                                    "se": [],
                                    "runs": []
                                })

                summary[dev_value]["subsidy"].append(sub_value)
                summary[dev_value]["mean"].append(mean)
                summary[dev_value]["se"].append(stderr)
                summary[dev_value]["runs"].append(results)

                print(f"dev={dev_value}, sub={sub_value} | mean={mean:.4f}, stderr={stderr:.4f}, n={n}")
    return summary

def compute_h2_market(snapshot, market, key="H2"):
    try:
        f_dict = snapshot["variables"]["f"]

        # Step 1: aggregate per scenario m
        h2_per_m = {}

        for (i, j, c, m), val in f_dict.items():

            if j == market and c == key:
                if val is not None and not np.isnan(val):
                    h2_per_m[m] = h2_per_m.get(m, 0.0) + val

        if not h2_per_m:
            return None

        values = list(h2_per_m.values())

        return np.mean(values), np.std(values), len(values)

    except KeyError:
        return None

def plot_hydrogen_consumption_by_market(base_folder, deviation, output_folder):
    """
    Plot hydrogen consumption differences per market using CRN.
    The first subsidy (subs[0]) is used as the base for each run.

    Args:
        base_folder: Path to the experiment folder
        deviation: Deviation value to analyze
        output_folder: Where to save the plot
    """
    os.makedirs(output_folder, exist_ok=True)

    base = Path(base_folder)
    dev_dir = base / f"dev{deviation}"

    # market -> subsidy -> list of run values
    market_consumption = {}

    subsidy_values = []

    # -------------------------------
    # Collect data
    # -------------------------------
    for sub_dir in sorted(dev_dir.glob("sub*/")):
        sub_value = float(sub_dir.name.replace("sub", ""))
        subsidy_values.append(sub_value)

        for run_dir in sorted(sub_dir.glob("run*/")):
            files = list(run_dir.glob("*.pkl"))
            if not files:
                continue

            snapshot = load_snapshot(files[0])

            for market in [ "DUNKERQUE", "ZEEBRUGGE",  "EMDEN", "EASINGTON", "DORNUM", "ST.FERGUS"]:
                market_consumption.setdefault(market, {}).setdefault(sub_value, [])
                value, std, n = compute_h2_market(snapshot, market, key = "H2")
                scaled_value = value * HYDROGEN_MSCM_MWH / 1000
                market_consumption[market][sub_value].append(scaled_value)

    # -------------------------------
    # Plot
    # -------------------------------
    plt.figure(figsize=(10, 5))

    for j, (market, sub_dict) in enumerate(sorted(market_consumption.items())):

        # Sort subsidies
        sorted_data = sorted(sub_dict.items())  # [(sub, [runs]), ...]

        subs = [x[0] for x in sorted_data]
        runs_per_sub = [x[1] for x in sorted_data]

        # Ensure consistent number of runs
        min_runs = min(len(r) for r in runs_per_sub)
        runs_per_sub = [r[:min_runs] for r in runs_per_sub]

        # Convert to array: shape (n_subs, n_runs)
        data = np.array(runs_per_sub)

        # -------------------------------
        # CRN: subtract base (subs[0]) per run
        # -------------------------------
        base = data[0, :]  # shape (n_runs,)
        diffs = data - base  # broadcasting

        # -------------------------------
        # Statistics
        # -------------------------------
        means = np.mean(diffs, axis=1)
        ses = np.std(diffs, axis=1, ddof=1) / np.sqrt(diffs.shape[1])

        color = PASTEL_COLORS[j % len(PASTEL_COLORS)]

        # Line
        plt.plot(subs, means, '-', label=f"Market {market}", color=color)

        # Error band
        lower = means - ses
        upper = means + ses
        plt.fill_between(subs, lower, upper, color=color, alpha=0.2)

    plt.xlabel('Subsidy (Euro/MWh)')
    plt.ylabel('Hydrogen Consumption (GWh)')
    plt.title(f'Hydrogen Consumption by Market (Deviation {int(deviation*100)}%)')
    plt.grid(alpha=0.3, axis="y")
    plt.legend()

    plt.savefig(os.path.join(output_folder, f"h2_consumption_by_market_dev{deviation}.png"))
    plt.close()


def plot_NG_consumption_by_market(base_folder, deviation, output_folder):
    """
    Plot natural gas consumption differences per market using CRN.
    The first subsidy (subs[0]) is used as the base for each run.

    Args:
        base_folder: Path to the experiment folder
        deviation: Deviation value to analyze
        output_folder: Where to save the plot
    """
    os.makedirs(output_folder, exist_ok=True)

    base = Path(base_folder)
    dev_dir = base / f"dev{deviation}"

    # market -> subsidy -> list of run values
    market_consumption = {}

    subsidy_values = []

    # -------------------------------
    # Collect data
    # -------------------------------
    for sub_dir in sorted(dev_dir.glob("sub*/")):
        sub_value = float(sub_dir.name.replace("sub", ""))
        subsidy_values.append(sub_value)

        for run_dir in sorted(sub_dir.glob("run*/")):
            files = list(run_dir.glob("*.pkl"))
            if not files:
                continue

            snapshot = load_snapshot(files[0])

            for market in ["DORNUM", "EMDEN", "ZEEBRUGGE", "DUNKERQUE", "EASINGTON", "ST.FERGUS"]:
                market_consumption.setdefault(market, {}).setdefault(sub_value, [])
                value, std, n = compute_h2_market(snapshot, market, key = "NG")  # Assuming a similar function for natural gas
                market_consumption[market][sub_value].append(value)

    # -------------------------------
    # Plot
    # -------------------------------
    plt.figure(figsize=(10, 5))

    for j, (market, sub_dict) in enumerate(sorted(market_consumption.items())):

        # Sort subsidies
        sorted_data = sorted(sub_dict.items())  # [(sub, [runs]), ...]

        subs = [x[0] for x in sorted_data]
        runs_per_sub = [x[1] for x in sorted_data]

        # Ensure consistent number of runs
        min_runs = min(len(r) for r in runs_per_sub)
        runs_per_sub = [r[:min_runs] for r in runs_per_sub]

        # Convert to array: shape (n_subs, n_runs)
        data = np.array(runs_per_sub)

        # -------------------------------
        # CRN: subtract base (subs[0]) per run
        # -------------------------------
        base = data[0, :]  # shape (n_runs,)
        diffs = data - base  # broadcasting

        # -------------------------------
        # Statistics
        # -------------------------------
        means = np.mean(diffs, axis=1)
        ses = np.std(diffs, axis=1, ddof=1) / np.sqrt(diffs.shape[1])

        color = PASTEL_COLORS[j % len(PASTEL_COLORS)]

        # Line
        plt.plot(subs, means, '-', label=f"Market {market}", color=color)

        # Error band
        lower = means - ses
        upper = means + ses
        plt.fill_between(subs, lower, upper, color=color, alpha=0.2)

    plt.xlabel('Subsidy (Euro/MWh)')
    plt.ylabel('Change in Natural Gas Consumption (Mscm)')
    plt.title(f'Natural Gas Consumption by Market (Deviation {int(deviation*100)}%)')
    plt.grid(alpha=0.3, axis="y")
    plt.legend()

    plt.savefig(os.path.join(output_folder, f"ng_consumption_by_market_dev{deviation}.png"))
    plt.close()

def plot_hydrogen_production_by_subsidy(h2_dict, folder):
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(10, 5))

    # Step 1: collect all unique subsidies
    all_subsidies = set()
    for stats in h2_dict.values():
        all_subsidies.update(stats["subsidy"])

    # Step 2: loop over each subsidy (these become the lines)
    for j, sub in enumerate(sorted(all_subsidies)):
        deviations = []
        means = []
        ses = []

        # Step 3: collect values across deviations
        for deviation, stats in sorted(h2_dict.items()):
            for s, m, se, runs in zip(stats["subsidy"], stats["mean"], stats["se"], stats["runs"]):
                if s == sub:
                    deviations.append(deviation)
                    means.append(m * HYDROGEN_MSCM_MWH /1000)
                    ses.append(se * HYDROGEN_MSCM_MWH /1000)

        # sort by deviation for clean lines
        sorted_data = sorted(zip(deviations, means, ses), key=lambda x: x[0])
        devs, means, ses = zip(*sorted_data)

        plt.errorbar(
            devs,
            means,
            yerr=ses,
            fmt='o-',
            capsize=5,
            label=f"Subsidy {sub}",
            color = PASTEL_COLORS[j % len(PASTEL_COLORS)]
        )

    plt.xlabel('Deviation')
    plt.ylabel('Average Hydrogen Production (Gwh)')
    plt.title('Average Hydrogen Production vs Deviation')
    plt.grid(alpha=0.3, axis="y")
    plt.legend(loc = "upper left")

    plt.savefig(os.path.join(folder, "hydrogen_production_vs_deviation.png"))
    plt.close()

def plot_hydrogen_production(h2_dict, folder): 
    os.makedirs(folder, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    
    for j, (label, stats) in enumerate(sorted(h2_dict.items())): 
        # sort by subsidy to ensure clean lines 
        sorted_data = sorted(
            zip(stats["subsidy"], stats["mean"], stats["se"], stats["runs"]), 
            key=lambda x: x[0]
        )
        subs, means, ses, runs = zip(*sorted_data)
        
        means = [m * HYDROGEN_MSCM_MWH/ 1000 for m in means]
        ses = [se * HYDROGEN_MSCM_MWH /1000 for se in ses]

        color = PASTEL_COLORS[j % len(PASTEL_COLORS)]
        
        # Line (no dots)
        plt.plot(subs, means, '-', label=f"Deviation {int(label *100)}%", color=color)
        
        # Shaded error band
        lower = [m - se for m, se in zip(means, ses)]
        upper = [m + se for m, se in zip(means, ses)]
        
        plt.fill_between(subs, lower, upper, color=color, alpha=0.2)
    
    plt.xlabel('Subsidy (Euro/MWh)') 
    plt.ylabel('Average Hydrogen Production (Gwh)')
    plt.title('Average Hydrogen Production vs Subsidy')
    plt.grid(alpha=0.3, axis="y")
    plt.legend() 
    
    plt.savefig(os.path.join(folder, "hydrogen_production_vs_subsidy.png")) 
    plt.close()

def plot_subsidy_cost(h2_dict, folder):
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(10, 5))

    for j, (label, stats) in enumerate(sorted(h2_dict.items())):
        # sort by subsidy to ensure clean lines
        sorted_data = sorted(
            zip(stats["subsidy"], stats["mean"], stats["se"]),
            key=lambda x: x[0]
        )

        subs, means, ses = zip(*sorted_data)

        costs = [mean * subs[i] * HYDROGEN_MSCM_MWH / 1000000 for i, mean in enumerate(means)] 
        cost_ses = [se * subs[i] * HYDROGEN_MSCM_MWH / 1000000 for i, se in enumerate(ses)]

        color = PASTEL_COLORS[j % len(PASTEL_COLORS)]

        # Line (no markers)
        plt.plot(subs, costs, '-', label=f"Deviation {int(label *100)}%", color=color)

        # Shaded band
        lower = [c - se for c, se in zip(costs, cost_ses)]
        upper = [c + se for c, se in zip(costs, cost_ses)]

        plt.fill_between(subs, lower, upper, color=color, alpha=0.2)

    plt.xlabel('Subsidy (Euro/MWh)')
    plt.ylabel('Average Subsidy cost (Million Euro)')
    plt.title('Average Subsidy cost vs Subsidy')
    plt.grid(alpha=0.3, axis="y")
    plt.legend()

    plt.savefig(os.path.join(folder, "hydrogen_production_vs_cost.png"))
    plt.close()


def analyze_objectives(base_folder):
    summary = {}

    base = Path(base_folder)
    print("Objective value:")
    for dev_dir in sorted(base.glob("dev*/")):
        dev_value = float(dev_dir.name.replace("dev", ""))

        for sub_dir in sorted(dev_dir.glob("sub*/")):
            sub_value = float(sub_dir.name.replace("sub", ""))
            results = []

            for run_dir in sorted(sub_dir.glob("run*/")):
                files = list(run_dir.glob("*.pkl"))
                if LOAD_ONE_RUN:
                    if run_dir.name != LOAD_ONE_RUN:
                        continue
                if not files:
                    continue

                with open(files[0], "rb") as f:
                    snapshot = pickle.load(f)

                val = compute_objective(snapshot)
                if val is not None:
                    results.append(val)

            if len(results) <= MINIMUM_RUNS-1:
                continue

            results = np.array(results)

            mean = np.mean(results)
            se = np.std(results, ddof=1) / np.sqrt(len(results))

            # store in structured format
            summary.setdefault(dev_value, {
                "subsidy": [],
                "mean": [],
                "se": [],
                "runs": []
            })

            summary[dev_value]["subsidy"].append(sub_value)
            summary[dev_value]["mean"].append(mean)
            summary[dev_value]["se"].append(se)
            summary[dev_value]["runs"].append(results)
            print(f"dev={dev_value}, sub={sub_value} | mean={mean:.4f}, stderr={se:.4f}, n={len(results)}")
    return summary

def plot_objective_values(objective_dict, folder):
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(10, 5))

    for j, (label, stats) in enumerate(sorted(objective_dict.items())):
        # ensure correct ordering by subsidy
        sorted_data = sorted(
            zip(stats["subsidy"], stats["mean"], stats["se"], stats["runs"]),
            key=lambda x: x[0]
        )

        subs, means, ses, runs = zip(*sorted_data)

        diffs = []
        base = runs[0]

        for r in runs:
            min_len = min(len(r), len(base))
            diffs.append([r[i] - base[i] for i in range(min_len)])

        diff_means = []
        diff_ses = []

        for d in diffs:
            d = np.array(d)
            n = len(d)

            mean = np.mean(d)
            se = np.std(d, ddof=1) / np.sqrt(n)

            diff_means.append(mean/ 1000000)
            diff_ses.append(se / 1000000)

        color = PASTEL_COLORS[j % len(PASTEL_COLORS)]

        # Line (no markers)
        plt.plot(subs, diff_means, '-', label=f"Deviation {int(label*100)}%", color=color)

        # Shaded uncertainty band
        lower = [m - se for m, se in zip(diff_means, diff_ses)]
        upper = [m + se for m, se in zip(diff_means, diff_ses)]

        plt.fill_between(subs, lower, upper, color=color, alpha=0.2)

    plt.xlabel('Subsidy (Euro/MWh)')
    plt.ylabel('Objective Value Effect (Million Euro)')
    plt.title('Objective Value Effect vs Subsidy')
    plt.grid(alpha=0.3, axis="y")
    plt.legend()

    plt.savefig(os.path.join(folder, "objective_vs_subsidy.png"))
    plt.close()


def plot_net_effect(objective_dict, h2_dict, folder, co2_method ="zero"):
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(10, 5))

    if co2_method == "zero":
        co2_savings_unit = 0
    elif co2_method == "energy":
        co2_cost_ng = 2.134 * 1000
        co2_cost_green_h2 = 1.080 / 11.94 * 1000
        
        conversion_ng = 39.8 / 3.6 * 1000
        conversion_h2 = 12.7 / 3.6 * 1000

        co2_savings_unit = 185 * (co2_cost_ng - co2_cost_green_h2) * (conversion_h2 / conversion_ng)
    elif co2_method == "volume":
        co2_cost_ng = 2.134 * 1000
        co2_cost_green_h2 = 1.080 / 11.94 * 1000
        co2_savings_unit = 185 * (co2_cost_ng - co2_cost_green_h2)
    else:
        raise Exception("No accepted co2 saving method")

    print("CO2 cost per Mcsm:", co2_savings_unit)

    for j, label in enumerate(sorted(objective_dict.keys())):
        obj_stats = objective_dict[label]
        h2_stats = h2_dict[label]

        sorted_data = sorted(
            zip(obj_stats["subsidy"], obj_stats["runs"], h2_stats["runs"]),
            key=lambda x: x[0]
        )

        subs, obj_runs, h2_runs = zip(*sorted_data)

        # STEP 1: compute net effect per run
        net_runs = []
        base_obj_runs = obj_runs[0]

        for s, r_obj, r_h2 in zip(subs, obj_runs, h2_runs):
            min_len = min(len(base_obj_runs), len(r_obj), len(r_h2))

            net_r = []
            for i in range(min_len):
                delta_obj = r_obj[i] - base_obj_runs[i]
                sub_cost = s * HYDROGEN_MSCM_MWH * r_h2[i]
                co2_savings = r_h2[i] * co2_savings_unit

                net = delta_obj - sub_cost + co2_savings
                net_r.append(net)

            net_runs.append(net_r)

        # STEP 2: CRN
        diffs = []
        base = net_runs[0]

        for r in net_runs:
            min_len = min(len(r), len(base))
            diffs.append([r[i] - base[i] for i in range(min_len)])

        diff_means = []
        diff_ses = []

        for d in diffs:
            d = np.array(d)
            n = len(d)

            mean = np.mean(d)
            se = np.std(d, ddof=1) / np.sqrt(n)

            diff_means.append(mean)
            diff_ses.append(se)

        color = PASTEL_COLORS[j % len(PASTEL_COLORS)]

        # Line (no markers)
        plt.plot(subs, diff_means, '-', label=f"Deviation {label* 100}%", color=color)

        # Shaded band
        lower = [m - se for m, se in zip(diff_means, diff_ses)]
        upper = [m + se for m, se in zip(diff_means, diff_ses)]

        plt.fill_between(subs, lower, upper, color=color, alpha=0.2)

    plt.xlabel('Subsidy (Euro/MWh)')
    plt.ylabel('Net Effect (Euro)')
    plt.title('Net Welfare Effect of Subsidy')
    plt.grid(alpha=0.3, axis="y")
    plt.legend()

    plt.savefig(os.path.join(folder, f"net_effect_vs_subsidy_{co2_method}.png"))
    plt.close()

        # Social cost savings per displaced mwh (transformed to mscm)
        # Sources:
        # Costs per kwh :   https://co2emissiefactoren.nl/ 
        # Conversion rates : https://www.engineeringtoolbox.com/fuels-higher-calorific-values-d_169.html
        # https://www-nature-com.vu-nl.idm.oclc.org/articles/s41586-022-05224-9  # social cost co2

        # Social cost savings per mscm
        # Sources:
        # https://h2tools.org/hyarc/calculator-tools/hydrogen-conversions-calculator # KG to scm transformation h2
        # https://co2emissiefactoren.nl/  Emissionfactors
        # https://www-nature-com.vu-nl.idm.oclc.org/articles/s41586-022-05224-9  # social cost co2

def plot_roi(objective_dict, h2_dict, folder, co2_method="zero"):
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(10, 5))

    if co2_method == "zero":
        co2_savings_unit = 0
    elif co2_method == "energy":
        co2_cost_ng = 2.134 * 1000
        co2_cost_green_h2 = 1.080 / 11.94 * 1000
        
        conversion_ng = 39.8 / 3.6 * 1000
        conversion_h2 = 12.7 / 3.6 * 1000

        co2_savings_unit = 185 * (co2_cost_ng - co2_cost_green_h2) * (conversion_h2 / conversion_ng)
    elif co2_method == "volume":
        co2_cost_ng = 2.134 * 1000
        co2_cost_green_h2 = 1.080 / 11.94 * 1000
        co2_savings_unit = 185 * (co2_cost_ng - co2_cost_green_h2)
    else:
        raise Exception("No accepted co2 saving method")

    print("CO2 cost per Mcsm:", co2_savings_unit)

    for j, label in enumerate(sorted(objective_dict.keys())):
        obj_stats = objective_dict[label]
        h2_stats = h2_dict[label]

        sorted_data = sorted(
            zip(
                obj_stats["subsidy"],
                obj_stats["runs"],
                h2_stats["runs"]
            ),
            key=lambda x: x[0]
        )

        subs, obj_runs, h2_runs = zip(*sorted_data)

        # STEP 1: compute ROI per run
        roi_runs = []
        base_obj_runs = obj_runs[0]

        for s, r_obj, r_h2 in zip(subs, obj_runs, h2_runs):
            min_len = min(len(base_obj_runs), len(r_obj), len(r_h2))

            roi_r = []
            for i in range(min_len):
                delta_obj = r_obj[i] - base_obj_runs[i]
                scaling_factor = 0
                sub_cost = s * HYDROGEN_MSCM_MWH * r_h2[i]
                co2_savings = r_h2[i] * co2_savings_unit

                if sub_cost > 0 and s >= 30:
                    roi = (delta_obj - sub_cost + co2_savings) / (sub_cost + scaling_factor)
                else:
                    roi = 0
                roi_r.append(roi)

            roi_runs.append(roi_r)

        # STEP 2: CRN differences
        diffs = []
        base = roi_runs[0]

        for r in roi_runs:
            min_len = min(len(r), len(base))
            diffs.append([r[i] - base[i] for i in range(min_len)])

        roi_means = []
        roi_ses = []

        for i, d in enumerate(diffs):
            if i < len(subs) and subs[i] < 30:
                continue
            
            d = np.array(d)
            n = len(d)

            mean = np.mean(d)
            se = np.std(d, ddof=1) / np.sqrt(n) if n > 1 else 0

            roi_means.append(mean * 100)
            roi_ses.append(se * 100) # convert to percentage for better readability

        # Match subs
        filtered_subs = [s for s in subs if s >= 30]

        color = PASTEL_COLORS[j % len(PASTEL_COLORS)]

        # Line (no markers)
        plt.plot(filtered_subs, roi_means, '-', label=f"Deviation {int(label * 100)}%", color=color)

        # Shaded band
        lower = [m - se for m, se in zip(roi_means, roi_ses)]
        upper = [m + se for m, se in zip(roi_means, roi_ses)]

        plt.fill_between(filtered_subs, lower, upper, color=color, alpha=0.2)

    plt.xlabel('Subsidy (Euro/MWh)')
    plt.ylabel('Return on investment (%)')
    plt.title('Return on investment of Subsidy')
    plt.grid(alpha=0.3, axis="y")
    plt.legend()

    plt.savefig(os.path.join(folder, f"roi_vs_subsidy_{co2_method}.png"))
    plt.close()

def plot_roi_cost(objective_dict, h2_dict, folder, co2_method="zero"):
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(10, 5))

    if co2_method == "zero":
        co2_savings_unit = 0
    elif co2_method == "energy":
        co2_cost_ng = 2.134 * 1000
        co2_cost_green_h2 = 1.080 / 11.94 * 1000
        
        conversion_ng = 39.8 / 3.6 * 1000
        conversion_h2 = 12.7 / 3.6 * 1000

        co2_savings_unit = 185 * (co2_cost_ng - co2_cost_green_h2) * (conversion_h2 / conversion_ng)
    elif co2_method == "volume":
        co2_cost_ng = 2.134 * 1000
        co2_cost_green_h2 = 1.080 / 11.94 * 1000
        co2_savings_unit = 185 * (co2_cost_ng - co2_cost_green_h2)
    else:
        raise Exception("No accepted co2 saving method")

    print("CO2 cost per Mcsm:", co2_savings_unit)

    for j, label in enumerate(sorted(objective_dict.keys())):
        obj_stats = objective_dict[label]
        h2_stats = h2_dict[label]

        sorted_data = sorted(
            zip(
                obj_stats["subsidy"],
                obj_stats["runs"],
                h2_stats["runs"]
            ),
            key=lambda x: x[0]
        )

        subs, obj_runs, h2_runs = zip(*sorted_data)

        # STEP 1: compute ROI per run
        roi_runs = []
        sub_costs = []
        base_obj_runs = obj_runs[0]

        for s, r_obj, r_h2 in zip(subs, obj_runs, h2_runs):
            min_len = min(len(base_obj_runs), len(r_obj), len(r_h2))

            roi_r = []
            sub_cost_list = []
            for i in range(min_len):
                delta_obj = r_obj[i] - base_obj_runs[i]
                scaling_factor = 0
                sub_cost = s * HYDROGEN_MSCM_MWH * r_h2[i]
                co2_savings = r_h2[i] * co2_savings_unit

                if sub_cost > 0 and s >= 30:
                    roi = (delta_obj - sub_cost + co2_savings) / (sub_cost + scaling_factor)
                else:
                    roi = 0
                roi_r.append(roi)
                sub_cost_list.append(sub_cost)

            roi_runs.append(roi_r)
            sub_costs.append(np.mean(sub_cost_list) / 1_000_000)  # Convert to Million Euro

        # STEP 2: CRN differences
        diffs = []
        base = roi_runs[0]

        for r in roi_runs:
            min_len = min(len(r), len(base))
            diffs.append([r[i] - base[i] for i in range(min_len)])

        roi_means = []
        roi_ses = []
        filtered_sub_costs = []

        for i, d in enumerate(diffs):
            if i < len(subs) and subs[i] < 30:
                continue
            
            d = np.array(d)
            n = len(d)

            mean = np.mean(d)
            se = np.std(d, ddof=1) / np.sqrt(n) if n > 1 else 0

            roi_means.append(mean * 100)
            roi_ses.append(se * 100)  # convert to percentage for better readability
            filtered_sub_costs.append(sub_costs[i])

        color = PASTEL_COLORS[j % len(PASTEL_COLORS)]

        # Line (no markers)
        plt.plot(filtered_sub_costs, roi_means, '-', label=f"Deviation {int(label * 100)}%", color=color)

        # Shaded band
        lower = [m - se for m, se in zip(roi_means, roi_ses)]
        upper = [m + se for m, se in zip(roi_means, roi_ses)]

        plt.fill_between(filtered_sub_costs, lower, upper, color=color, alpha=0.2)

    plt.xlabel('Subsidy Cost (Million Euro)')
    plt.ylabel('Return on investment (%)')
    plt.title('Return on investment of Subsidy')
    plt.grid(alpha=0.3, axis="y")
    plt.legend()

    plt.savefig(os.path.join(folder, f"roi_vs_cost_{co2_method}.png"))
    plt.close()


def analyze_network_flows(subsidy, deviation, base_folder):
    """
    Extract all flow variables for a given (subsidy, deviation) setting.

    Returns:
        dict with:
            - "flows": list of flow dictionaries per run
            - "mean": aggregated mean flow (per variable)
            - "se": standard error (per variable)
    """
    base = Path(base_folder)

    dev_dir = base / f"dev{deviation}"
    sub_dir = dev_dir / f"sub{subsidy}"
    flows = []

    for run_dir in sorted(sub_dir.glob("run*/")):
        files = list(run_dir.glob("*.pkl"))
        if not files:
            continue

        with open(files[0], "rb") as f:
            snapshot = pickle.load(f)

        # ---- Extract flows from snapshot ----
        # This assumes your snapshot stores flows in model.f as before
        flow_dict = {}

        f_vals = snapshot["variables"]["f"]
        sp = snapshot["parameters"]["sp"]

        for (a_in, a_out, c, s), val in f_vals.items():
            prob = sp[3,s]

            if (a_in, a_out) not in flow_dict:
                flow_dict[a_in, a_out] = {}

            if c not in flow_dict[a_in, a_out]:
                flow_dict[a_in, a_out][c] = 0.0

            flow_dict[a_in, a_out][c] += prob * val
        flows.append(flow_dict)
        # print(f"Extracted flow for run {run_dir.name}: {flow_dict}")

    if len(flows) == 0:
        return None

    # ---- Aggregate (example: mean & SE per variable/index) ----
    # Convert to structured numeric arrays
    aggregated = {}

    # Collect all variable names
    var_names = flows[0].keys()

    for vname in var_names:
        # collect all runs for this variable
        values = []

        for flow in flows:
            if isinstance(flow[vname], dict):
                values.append(list(flow[vname].values()))
            else:
                values.append([flow[vname]])

        values = np.array(values)

        mean = np.mean(values, axis=0)
        se = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])

        aggregated[vname] = {
            "mean": mean,
            "se": se
        }

        #print(f"Aggregated flow for variable {vname}: mean={mean}, se={se}")

    return {
        "flows": flows,
        "aggregated": aggregated
    }


def find_node_by_coords(node_rows, coords, tol=1e-6):
    try:
        coords = ast.literal_eval(coords)
    except (ValueError, SyntaxError):
        return coords

    x, y = coords
    matches = node_rows[
        node_rows.geometry.apply(
            lambda g: abs(g.x - x) < tol and abs(g.y - y) < tol
        )
    ]
    if len(matches) == 0:
        raise ValueError(f"No node found near {coords}")
    if len(matches) > 1:
        raise ValueError(f"Multiple nodes found near {coords}")
    return matches.iloc[0]["location"]

def network_plot_hydrogen_production(subsidy, deviation, base_folder, output_folder):
    # -------------------------------
    # Load flows
    # -------------------------------
    result = analyze_network_flows(subsidy, deviation, base_folder)
    if result is None:
        print("No flow data found.")
        return None

    # ---- Average flows over runs ----
    avg_flow = result["aggregated"]

    # -------------------------------
    # Load geo data
    # -------------------------------
    FOLDER = "data/data_analysis_results/Geojson_pipelines/"
    input_path = FOLDER + "study_case_network.geojson"
    os.makedirs(output_folder, exist_ok=True)
    output_html = os.path.join(output_folder, f"network_h2_sub{subsidy}_dev{deviation}.html")

    gdf = gpd.read_file(input_path)

    nodes = gdf[gdf.geometry.type == "Point"].copy()
    edges = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()

    # -------------------------------
    # Create map
    # -------------------------------
    center = gdf.geometry.union_all().centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=7, tiles="CartoDB positron")

    # -------------------------------
    # Helpers
    # -------------------------------
    def is_nan_safe(val):
        if isinstance(val, numbers.Number):
            return np.isnan(val)
        if isinstance(val, pd.Timestamp):
            return not pd.isna(val)
        return False

    def make_tooltip(row):
        props = []
        for col in row.index:
            if col in ["pipName", "fromFacili", "toFacility", "max_flow", "location", "node_type"]:
                val = row[col]
                if val is not None and not is_nan_safe(val):
                    props.append(f"<b>{col}</b>: {val}")
        return "<br>".join(props)

    # -------------------------------
    # Determine max H2 flow (for scaling)
    # -------------------------------
    h2_values = []
    tot_flow_values = []
    for arc in avg_flow.keys():
        h2_values.append(avg_flow[arc]["mean"][2])
        tot_flow_values.append(sum(avg_flow[arc]["mean"]))

    #max_flow = 5 #max(h2_values) if h2_values else 1.0
    max_tot_flow = 30 #max(tot_flow_values) if tot_flow_values else 1.0



    # -------------------------------
    # Add edges (WITH hydrogen coloring)
    # -------------------------------
    for _, row in edges.iterrows():
        geom = row.geometry
        lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]

        # ---- Map arc ----
        a_in = find_node_by_coords(nodes, row.get("from_node"))
        a_out = find_node_by_coords(nodes, row.get("to_node"))

        h2_flow = avg_flow[a_in, a_out]["mean"][2]
        total_flow = sum(avg_flow[a_in, a_out]["mean"])

        # Normalize
        intensity = h2_flow / total_flow if total_flow > 0 else 0 # max_flow

        # Blue gradient (light → strong blue)
        def interpolate_color(intensity, start=(60, 60, 60), end=(0, 201, 255)):
            intensity = intensity * 5 # Max 20% accross pipelines
            r = int(start[0] + (end[0] - start[0]) * intensity)
            g = int(start[1] + (end[1] - start[1]) * intensity)
            b = int(start[2] + (end[2] - start[2]) * intensity)
            return f"rgb({r}, {g}, {b})"

        def adjust_intensity(intensity):
            return np.power(intensity, 0.8)  # Adjust exponent for better contrast

        color_intensity = adjust_intensity(intensity)
        color = interpolate_color(color_intensity)

        # Thickness scaling
        weight_intensity = total_flow / max_tot_flow if max_tot_flow > 0 else 0
        weight = 2 + 6 * np.power(weight_intensity, 0.3)
        

        for line in lines:
            coords = [(lat, lon) for lon, lat in line.coords]

            tooltip = make_tooltip(row) + f"<br><b>H2 flow</b>: {h2_flow:.2f}" + f"<br><b>Total flow</b>: {total_flow:.2f}"

            folium.PolyLine(
                coords,
                color=color,
                weight=weight,
                opacity=0.85,
                tooltip=folium.Tooltip(tooltip, sticky=True),
            ).add_to(m)

    # -------------------------------
    # Add nodes (unchanged)
    # -------------------------------
    for _, row in nodes.iterrows():
        point = row.geometry
        tooltip = make_tooltip(row)

        color_map = {
            "Generation": "#FFB3BA",
            "Processing": "#BAFFC9",
            "Market": "#BAE1FF",
            "Compression": "#FFFFBA",
            "Junction": "#E0BBE4",
        }

        node_type = row.get("node_type", "junction")
        color = color_map.get(node_type, "#CCCCCC")

        folium.CircleMarker(
            location=[point.y, point.x],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            tooltip=folium.Tooltip(tooltip, sticky=True),
        ).add_to(m)

    # -------------------------------
    # Add legend (with hydrogen)
    # -------------------------------
    # Precompute example values
    flow_ticks = [0, 0.25, 0.5, 0.75, 1.0]
    flow_values = [f"{t * max_tot_flow:.1f}" for t in flow_ticks]

    legend_html = f'''
        <div style="position: fixed; 
            bottom: 50px; right: 50px; width: 260px; height: auto; 
            background-color: white; border:2px solid grey; z-index:9999; 
            font-size:14px; padding: 12px; overflow-y: auto;">

        <p style="margin: 0 0 10px 0; font-weight: bold;">Legend</p>

        <p><b>Node Types</b></p>
        <p><span style="background-color: #FFB3BA; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></span> Generation</p>
        <p><span style="background-color: #BAFFC9; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></span> Processing</p>
        <p><span style="background-color: #BAE1FF; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></span> Market</p>
        <p><span style="background-color: #FFFFBA; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></span> Compression</p>
        <p><span style="background-color: #E0BBE4; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></span> Junction</p>

        <p><b>Hydrogen (%)</b></p>
        <div style="width: 100%; height: 15px; 
            background: linear-gradient(to right, 
            {interpolate_color(adjust_intensity(0))}, 
            {interpolate_color(adjust_intensity(0.05))}, 
            {interpolate_color(adjust_intensity(0.1))}, 
            {interpolate_color(adjust_intensity(0.15))}, 
            {interpolate_color(adjust_intensity(0.2))}); 
            border: 1px solid black;">
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>0</span>
            <span>10</span>
            <span>20</span>
        </div>

        <p><b>Flow (Mscm)</b></p>

        <div style="width: 100%; height: 20px; position: relative;">

            <!-- Tapered thickness bar -->
            <div style="
                width: 100%; 
                height: 100%;
                background: linear-gradient(to right,
                    rgba(0,0,0,0.2) 0%,
                    rgba(0,0,0,0.4) 25%,
                    rgba(0,0,0,0.6) 50%,
                    rgba(0,0,0,0.8) 75%,
                    rgba(0,0,0,1.0) 100%);
                clip-path: polygon(
                    0% 50%, 
                    100% 0%, 
                    100% 100%
                );
            "></div>

        </div>

        <div style="display: flex; justify-content: space-between; font-size: 12px;">
            <span>{flow_values[0]}</span>
            <span>{flow_values[2]}</span>
            <span>{flow_values[4]}</span>
        </div>
        </div>
        '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # -------------------------------
    # Save
    # -------------------------------
    m.save(output_html)
    print(f"Saved interactive map to: {output_html}")

    return m


def compute_overprovision(snapshot):
    """
    Compute entry and exit overprovision values from a snapshot.

    Returns:
        entry_overprovisions (list), exit_overprovisions (list)
    """

    f_vals = snapshot["variables"]["f"]
    entry_vals = snapshot["variables"]["x_entry"]
    exit_vals = snapshot["variables"]["x_exit"]

    # =========================
    # Total capacity per node
    # =========================
    entry_overprovision = []
    exit_overprovision = []

    M_3 = set(s for (a_in, a_out, c, s) in f_vals.keys())
    K = [1,2,3]

    N = set(n for (n, k, s) in entry_vals.keys())

    for s in M_3:
        entry_overprovision_nodes = 0
        exit_overprovision_nodes = 0
        entry_amount = 0
        exit_amount = 0

        for n in N:

            entry_val = 0
            exit_val = 0

            for k in K:
                if k ==1:
                    s_prime =1
                elif k ==2:
                    s_prime = (s-1) // 8 +1
                else:
                    s_prime = s
                # average over scenarios of that stage
                entry_val += entry_vals[n, k, s_prime]
                exit_val += exit_vals[n, k, s_prime]

            inflow = sum(val for (a_in, a_out, c, s_prime), val in f_vals.items() if a_out == n and s_prime == s )
            outflow = sum(val for (a_in, a_out, c, s_prime), val in f_vals.items() if a_in == n and s_prime == s )


            net_flow = inflow - outflow
            # if abs(net_flow) > 1:
            #     print(n, net_flow)


            if net_flow >0.01:
                e_overprovision = entry_val
                ex_overprovision = max(exit_val - net_flow, 0)
                # print(n, e_overprovision)
                # print(n, ex_overprovision)
            elif net_flow < -0.01:
                e_overprovision = max(entry_val + net_flow, 0)
                ex_overprovision = exit_val
                # print(n, e_overprovision)
                # print(n, ex_overprovision)
            else:
                e_overprovision = 0
                ex_overprovision = 0
                entry_val =0 
                ex_val = 0

            
            entry_overprovision_nodes += e_overprovision
            exit_overprovision_nodes += ex_overprovision
            entry_amount += entry_val
            exit_amount += exit_val
        
        entry_overprovision.append(entry_overprovision_nodes/ entry_amount if entry_amount > 0 else 0)
        exit_overprovision.append(exit_overprovision_nodes/ exit_amount if exit_amount > 0 else 0)

    return entry_overprovision, exit_overprovision

def analyze_overprovision(base_folder, deviation, subsidy):
    base = Path(base_folder)

    dev_dir = base / f"dev{deviation}"
    sub_dir = dev_dir / f"sub{subsidy}"
    print(sub_dir)
    entry_overprovisions = []
    exit_overprovisions = []

    for run_dir in sorted(sub_dir.glob("run*/")):
        files = list(run_dir.glob("*.pkl"))
        if not files:
            continue

        with open(files[0], "rb") as f:
            snapshot = pickle.load(f)
            entry_overprovision, exit_overprovision = compute_overprovision(snapshot)
            entry_overprovisions.append(np.mean(entry_overprovision))
            exit_overprovisions.append(np.mean(exit_overprovision))

    return entry_overprovisions, exit_overprovisions


def adjust_color(color, factor=0.7):
    """
    Darken (<1) or lighten (>1) a color.
    """
    rgb = np.array(mcolors.to_rgb(color))
    if factor < 1:
        return tuple(rgb * factor)  # darken
    else:
        return tuple(1 - (1 - rgb) / factor)  # lighten


def plot_overprovision_distribution(base_folder, deviation, subsidies, folder="figures/", show=False):
    """
    Create violin plots for entry and exit overprovisions across subsidies.

    Args:
        base_folder: path to experiment folder
        deviation: deviation value
        subsidies: list of subsidy values
        folder: output folder
        show: whether to display plot
    """

    # =========================
    # Collect data
    # =========================
    arc_data_selected = []

    for sub in subsidies:
        entry, exit_ = analyze_overprovision(base_folder, deviation, sub)

        entry = [e *100 for e in entry]
        exit_ = [e * 100 for e in exit_]

        arc_data_selected.append({
            "label": f"{int(sub)}",
            "entry": entry,
            "exit": exit_
        })

    # =========================
    # Prepare plotting data
    # =========================
    data = []
    labels = []
    positions = []

    pos = 1
    spacing = 1.5

    for d in arc_data_selected:
        # Entry (inlet)
        data.append(d["entry"])
        labels.append(f"{d['label']} \n (Entry)")
        positions.append(pos)

        # Exit (outlet)
        data.append(d["exit"])
        labels.append(f"{d['label']} \n (Exit)")
        positions.append(pos + 0.6)

        pos += spacing

    # =========================
    # Plot vertical violins
    # =========================
    fig, ax = plt.subplots(figsize=(10, 6))


    parts = ax.violinplot(
        data,
        positions=positions,
        vert=True,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    # =========================
    # Styling
    # =========================
    for i, pc in enumerate(parts['bodies']):
        color_idx = i % 2
        pc.set_facecolor(PASTEL_COLORS[color_idx % len(PASTEL_COLORS)])
        pc.set_edgecolor(PASTEL_COLORS[color_idx % len(PASTEL_COLORS)])
        pc.set_alpha(0.7)

    # =========================
    # Overlay TRUE data points
    # =========================
    for i, vals in enumerate(data):
        vals = np.array(vals)

        if len(vals) == 0:
            continue

        sorted_vals = np.sort(vals)
        unique_vals, counts = np.unique(sorted_vals, return_counts=True)

        x_positions = []
        y_positions = []

        for val, count in zip(unique_vals, counts):
            offsets = [0] if count == 1 else np.linspace(-0.08, 0.08, count)
            for offset in offsets:
                x_positions.append(positions[i] + offset)
                y_positions.append(val)

        base_color = PASTEL_COLORS[i % 2]
        point_color = adjust_color(base_color, factor=0.9)

        ax.scatter(
            x_positions,
            y_positions,
            s=14,
            alpha=1,
            color=point_color,
            edgecolors='none'
        )

    # =========================
    # Labels and layout
    # =========================
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(alpha=0.3, axis="y")

    ax.set_xlabel("Subsidy (Euro/Mwh)")
    ax.set_ylabel("Overprovision (%)")
    ax.set_title(f"Entry and Exit Overprovision Distributions (Deviation: {int(deviation*100)}%)")

    plt.tight_layout()
    Path(folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{folder}/overprovision_violin_dev{deviation}.png")

    if show:
        plt.show()

    plt.close(fig)

def plot_overprovision_bar(base_folder, deviation, subsidies, folder="figures/", show=False):
    """
    Create bar plot (mean + std) for entry and exit overprovisions across subsidies.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    entry_means, exit_means = [], []
    entry_stds, exit_stds = [], []

    # =========================
    # Collect + summarize data
    # =========================
    for sub in subsidies:
        entry, exit_ = analyze_overprovision(base_folder, deviation, sub)

        entry = np.array(entry) * 100
        exit_ = np.array(exit_) * 100

        entry_means.append(np.mean(entry) if len(entry) else 0)
        exit_means.append(np.mean(exit_) if len(exit_) else 0)

        entry_stds.append(np.std(entry) if len(entry) else 0)
        exit_stds.append(np.std(exit_) if len(exit_) else 0)

    # =========================
    # Plot
    # =========================
    x = np.arange(len(subsidies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Standard errors
    entry_ses = [std / np.sqrt(3) for std in entry_stds]
    exit_ses  = [std / np.sqrt(3) for std in exit_stds]

    # ---- ENTRY ----
    # Main bar (add outline)
    ax.bar(
        x - width/2,
        entry_means,
        width,
        label="Entry Capacity",
        color=PASTEL_COLORS[0],
        alpha=1,
        edgecolor=adjust_color(PASTEL_COLORS[0], factor=0.6),
        linewidth=1.0
    )

    # Error band (fill)
    entry_bottom = [m - se for m, se in zip(entry_means, entry_ses)]
    entry_top    = [m + se for m, se in zip(entry_means, entry_ses)]

    ax.bar(
        x - width/2,
        [2 * se for se in entry_ses],
        width,
        bottom=entry_bottom,
        color=adjust_color(PASTEL_COLORS[0], factor=1.15),
        alpha=0.2,
        edgecolor='none'
    )

    # Dashed top & bottom خطوط
    for xi, low, high in zip(x - width/2, entry_bottom, entry_top):
        ax.hlines(high, xi - width/2, xi + width/2,
                colors=adjust_color(PASTEL_COLORS[0], factor=0.6),
                linestyles='dashed', linewidth=1)
        ax.hlines(low, xi - width/2, xi + width/2,
                colors=adjust_color(PASTEL_COLORS[0], factor=0.6),
                linestyles='dashed', linewidth=1)


    # ---- EXIT ----
    # Main bar (add outline)
    ax.bar(
        x + width/2,
        exit_means,
        width,
        label="Exit Capacity",
        color=PASTEL_COLORS[1],
        alpha=1,
        edgecolor=adjust_color(PASTEL_COLORS[1], factor=0.6),
        linewidth=1.0
    )

    # Error band (fill)
    exit_bottom = [m - se for m, se in zip(exit_means, exit_ses)]
    exit_top    = [m + se for m, se in zip(exit_means, exit_ses)]

    ax.bar(
        x + width/2,
        [2 * se for se in exit_ses],
        width,
        bottom=exit_bottom,
        color=adjust_color(PASTEL_COLORS[1], factor=1.15),
        alpha=0.2,
        edgecolor='none'
    )

    # Dashed top & bottom lines
    for xi, low, high in zip(x + width/2, exit_bottom, exit_top):
        ax.hlines(high, xi - width/2, xi + width/2,
                colors=adjust_color(PASTEL_COLORS[1], factor=0.6),
                linestyles='dashed', linewidth=1)
        ax.hlines(low, xi - width/2, xi + width/2,
                colors=adjust_color(PASTEL_COLORS[1], factor=0.6),
                linestyles='dashed', linewidth=1)

    # =========================
    # Labels & layout
    # =========================
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(s)}" for s in subsidies])

    ax.set_xlabel("Subsidy (Euro/MWh)")
    ax.set_ylabel("Overprovision (%)")
    ax.set_title(f"Average Overprovision (Deviation: {int(deviation*100)}%)")

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{folder}/overprovision_bar_dev{deviation}.png")

    if show:
        plt.show()

    plt.close(fig)
    

# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    base_folder = f"study_case_model/figures/subsidy_experiment/{EXPERIMENT}/"
    sub_values = [0.0, 30.0, 45.0, 70.0]
    dev_values = [0.0, 0.1, 1.0]

    for sub_value in sub_values:
        for dev_value in dev_values:
            network_plot_hydrogen_production(sub_value, dev_value, base_folder, f"study_case_model/figures/subsidy_experiment/combined_results/html_networks/{EXPERIMENT}/")

    results = analyze_experiment(base_folder)

    if LOAD_ONE_RUN:
        plot_hydrogen_production(results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}")
        plot_hydrogen_production_by_subsidy(results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}")
        plot_hydrogen_consumption_by_market(base_folder,  dev_values[2], f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}")
        plot_NG_consumption_by_market(base_folder,  dev_values[2], f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}")
        plot_subsidy_cost(results, folder =f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}")
    else:
        plot_hydrogen_production(results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/")
        plot_hydrogen_production_by_subsidy(results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/")
        plot_hydrogen_consumption_by_market(base_folder,  dev_values[2], "study_case_model/figures/subsidy_experiment/combined_results/")
        plot_NG_consumption_by_market(base_folder,  dev_values[2], f"study_case_model/figures/subsidy_experiment/combined_results/")
        plot_subsidy_cost(results, folder =f"study_case_model/figures/subsidy_experiment/combined_results/")

    objective_dict = analyze_objectives(base_folder)

    if LOAD_ONE_RUN:
        plot_objective_values(objective_dict, folder=f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}")
        plot_net_effect(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}")
        plot_net_effect(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}", co2_method="energy")
        plot_net_effect(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}", co2_method="volume")
        plot_roi(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}")
        plot_roi(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}", co2_method="energy")
        plot_roi(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}", co2_method="volume")
    else:
        plot_objective_values(objective_dict, folder=f"study_case_model/figures/subsidy_experiment/combined_results/")
        plot_net_effect(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/")
        plot_net_effect(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/", co2_method="energy")
        plot_net_effect(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/", co2_method="volume")
        plot_roi(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/")
        plot_roi(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/", co2_method="energy")
        plot_roi(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/", co2_method="volume")

    sub_values = [0.0, 25.0, 30.0, 36.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]
    #sub_values = [0.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    dev = 0.0
    if LOAD_ONE_RUN:
        #plot_overprovision_distribution(base_folder,deviation=0.0,subsidies= sub_values, folder=f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}")
        plot_overprovision_bar(base_folder,deviation=dev,subsidies= sub_values, folder=f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}")
    else:
        #plot_overprovision_distribution(base_folder,deviation=0.0,subsidies= sub_values, folder=f"study_case_model/figures/subsidy_experiment/combined_results/")
        plot_overprovision_bar(base_folder,deviation=dev,subsidies= sub_values, folder=f"study_case_model/figures/subsidy_experiment/combined_results/")