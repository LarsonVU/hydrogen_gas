import os
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os

LOAD_ONE_RUN = "run0"

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
            print(f"dev={dev_value}, sub={sub_value} | mean={mean:.4f}, stderr={se:.4f}, n={len(results)}")
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

        ses  = [np.sqrt(s**2 + ses[0]**2) for s in ses]
        means = means - means[0]

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

def plot_net_effect(objective_dict, h2_dict, folder, co2_method ="zero"):
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(10, 5))

    if co2_method == "zero":
        # No social cost on co2
        co2_savings_unit = 0
    elif co2_method == "energy":
        # Social cost savings per displaced mwh (transformed to mscm)
        # Sources:
        # Costs per kwh :   https://co2emissiefactoren.nl/ 
        # Conversion rates : https://www.engineeringtoolbox.com/fuels-higher-calorific-values-d_169.html
        # https://www-nature-com.vu-nl.idm.oclc.org/articles/s41586-022-05224-9  # social cost co2

        co2_cost_ng = 2.134 * 1000 # in tonne (metric ton) CO2e per Mscm 
        co2_cost_green_h2 = 1.080 /11.94 *1000 # in tonne (metric ton) CO2e per Mscm 
        
        conversion_ng = 39.8 /3.6 *1000 # mwh to Mscm
        conversion_h2 = 12.7 /3.6 *1000 #  mwh to Mscm

        co2_savings_unit = 185 * (co2_cost_ng-co2_cost_green_h2) * (conversion_h2/ conversion_ng) # Euro per Mscm
    elif co2_method == "volume":
        # Social cost savings per mscm
        # Sources:
        # https://h2tools.org/hyarc/calculator-tools/hydrogen-conversions-calculator # KG to scm transformation h2
        # https://co2emissiefactoren.nl/  Emissionfactors
        # https://www-nature-com.vu-nl.idm.oclc.org/articles/s41586-022-05224-9  # social cost co2 
        co2_cost_ng = 2.134 * 1000 # in tonne (metric ton) CO2e per Mscm 
        co2_cost_green_h2 = 1.080 /11.94 *1000 # in tonne (metric ton) CO2e per Mscm 
        co2_savings_unit = 185 * (co2_cost_ng -co2_cost_green_h2) # in euro per Mscm
    else:
        Exception("No accepted co2 saving method")
    print("CO2 cost per Mcsm:", co2_savings_unit)


    for label in sorted(objective_dict.keys()):
        obj_stats = objective_dict[label]
        h2_stats = h2_dict[label]

        # sort both consistently
        sorted_data = sorted(
            zip(obj_stats["subsidy"], obj_stats["mean"], obj_stats["se"],
                h2_stats["mean"], h2_stats["se"]),
            key=lambda x: x[0]
        )

        subs, obj_means, obj_ses, h2_means, h2_ses = zip(*sorted_data)
        base_obj = obj_means[0]

        net_means = []
        net_ses = []



        for s, m_obj, se_obj, m_h2, se_h2 in zip(subs, obj_means, obj_ses, h2_means, h2_ses):
            delta_obj =  m_obj - base_obj

            # correct net effect
            net = delta_obj - s *2.78 * 1000 * m_h2 + m_h2 *co2_savings_unit
            net_means.append(net)

            # error propagation (approx)
            var = se_obj**2 + obj_ses[0]**2 + (s**2) * (se_h2**2)
            net_ses.append(np.sqrt(var))

        plt.errorbar(
            subs,
            net_means,
            yerr=net_ses,
            fmt='o-',
            capsize=5,
            label=f"Deviation {label}"
        )

    plt.xlabel('Subsidy (Euro/MWh)')
    plt.ylabel('Net Effect (Euro)')
    plt.title('Net Welfare Effect of Subsidy')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(folder, f"net_effect_vs_subsidy_{co2_method}.png"))
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
    print(dev_dir, sub_dir)
    flows = []

    for run_dir in sorted(sub_dir.glob("run*/")):
        files = list(run_dir.glob("*.pkl"))
        print(files)
        if not files:
            continue

        with open(files[0], "rb") as f:
            snapshot = pickle.load(f)

        # ---- Extract flows from snapshot ----
        # This assumes your snapshot stores flows in model.f as before
        flow_dict = {}

        f_vals = snapshot["variables"]["f"]
        sp = snapshot["parameters"]["sp"]
        print(sp.items())
        for (a_in, a_out, c, s), val in f_vals.items():
            prob = sp[3,s]

            if (a_in, a_out) not in flow_dict:
                flow_dict[a_in, a_out] = {}

            if c not in flow_dict[a_in, a_out]:
                flow_dict[a_in, a_out][c] = 0.0

            flow_dict[a_in, a_out][c] += prob * val
            print(a_in, a_out, c, flow_dict[a_in, a_out][c])
        flows.append(flow_dict)

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

    return {
        "flows": flows,
        "aggregated": aggregated
    }


def network_plot_hydrogen_production(subsidy, deviation, folder):
    return None


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    base_folder = "study_case_model/figures/subsidy_experiment/run_27326/"
    # analyze_network_flows("60.0", "0.0", base_folder)
    results = analyze_experiment(base_folder)

    if LOAD_ONE_RUN:
        plot_hydrogen_production(results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}")
    else:
        plot_hydrogen_production(results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/")

    objective_dict = analyze_objectives(base_folder)

    if LOAD_ONE_RUN:
        plot_objective_values(objective_dict, folder=f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}")
        plot_net_effect(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}")
        plot_net_effect(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}", co2_method="energy")
        plot_net_effect(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}", co2_method="volume")
    else:
        plot_objective_values(objective_dict, folder=f"study_case_model/figures/subsidy_experiment/combined_results/")
        plot_net_effect(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/")
        plot_net_effect(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/", co2_method="energy")
        plot_net_effect(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/", co2_method="volume")