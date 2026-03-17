import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import sys
import os
import argparse
import csv
from collections import defaultdict

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import study_case_stochastic_model as scsm
import study_case_problem_file as scpf

# =========================
# Argument parsing
# =========================
parser = argparse.ArgumentParser()

parser.add_argument("--")
parser.add_argument("--amount_per_point", type=int, default=2)
parser.add_argument("--branches_stage2", type=int, default=2)
parser.add_argument("--branches_stage3", type=int, default=2)
parser.add_argument("--subsidies", type=float, nargs="+", default=[0, 40, 80])
parser.add_argument("--deviations", type=float, nargs="+", default=[0])
parser.add_argument("--upper_bounds", type=int, default=1)

parser.add_argument("--time_limit", type=float, default=None)

parser.add_argument("--data_folder", type=str,
                    default="study_case_model/scenario_variables/examine_subsidies/")
parser.add_argument("--figures_folder", type=str,
                    default="study_case_model/figures/examine_subsidies/")

parser.add_argument("--replot_only", action="store_true")



args = parser.parse_args()

# =========================
# Constants
# =========================
NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {
    1: 1,
    2: args.branches_stage2,
    3: args.branches_stage3
}
UPPER_BOUNDS = args.upper_bounds


DATA_FOLDER = args.data_folder
FIGURES_FOLDER = args.figures_folder

# =========================
# Core functions
# =========================
def change_hydrogen_price(G, value_range, amount_per_point, variable_name="unknown"):
    G_list = []
    for var in value_range:
        G_list_part = []
        for _ in range(amount_per_point):
            G_changed = G.copy()
            for node in G.nodes:
                if G.nodes[node][variable_name] is not None:
                    if G.nodes[node]["component_ratio"]["H2"] > 0:
                        G_changed.nodes[node][variable_name] = (
                            float(G.nodes[node][variable_name]) - var
                        )
            G_list_part.append(G_changed)
        G_list.append(G_list_part)
    return G_list


def create_scenario_trees(G_list, dev):
    scenario_list = []
    for i, G_part in enumerate(G_list):
        s_part = []
        for j, G in enumerate(G_part):
            folder = DATA_FOLDER + f"deviation{dev}/{i}/{j}/"
            s_part.append(
                scpf.create_scenarios(
                    NUMBER_OF_STAGES,
                    BRANCHES_PER_STAGE,
                    G,
                    folder=folder
                )
            )
        scenario_list.append(s_part)
    return scenario_list


def solve_multiple_problems(G, s_list, i, verbose=False, time_limit=None, allowed_deviation=0):
    result_list = []
    avg_hydrogen_production = []

    for j, scenarios in enumerate(s_list):
        print("run", j)

        model = scsm.create_model(G, scenarios, allowed_deviation=allowed_deviation, number_of_density_bounds=UPPER_BOUNDS)
        _ = scsm.solve_model(model, verbose, time_limit, precision=0.01)

        folder = DATA_FOLDER + f"deviation{allowed_deviation}/subsidy{i}/run{j}/"
        os.makedirs(folder, exist_ok=True)

        filename = folder + "model_snapshot.pkl"
        scsm.save_model_values(model, filename)

        result_list.append(pyo.value(model.objective))

        values = [pyo.value(model.h2_production[i]) for i in model.h2_production]
        avg_hydrogen_production.append(np.mean(values))

    return result_list, avg_hydrogen_production


def results_from_subsidy(G, s_super_list, subsidies, verbose=False, time_limit=None, deviation=0):
    objective_means = []
    objective_se = []

    hydrogen_means = []
    hydrogen_se = []

    for s, s_part in zip(subsidies, s_super_list):
        print("subsidy", s)

        results, avg_hydrogen_production = solve_multiple_problems(
            G, s_part, s, verbose, time_limit, allowed_deviation=deviation
        )

        n = len(results)

        objective_means.append(np.mean(results))
        objective_se.append(np.std(results) / np.sqrt(n))

        hydrogen_means.append(np.mean(avg_hydrogen_production))
        hydrogen_se.append(np.std(avg_hydrogen_production) / np.sqrt(n))

    return objective_means, objective_se, hydrogen_means, hydrogen_se


# =========================
# CSV SAVE / LOAD
# =========================
def save_results_to_csv(obj_dict, h2_dict, subsidies, folder):
    os.makedirs(folder, exist_ok=True)

    obj_file = os.path.join(folder, "objective_results.csv")
    h2_file = os.path.join(folder, "hydrogen_results.csv")

    with open(obj_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["deviation", "subsidy", "mean", "se"])
        for dev, stats in obj_dict.items():
            for s, mean, se in zip(subsidies, stats["mean"], stats["se"]):
                writer.writerow([dev, s, mean, se])

    with open(h2_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["deviation", "subsidy", "mean", "se"])
        for dev, stats in h2_dict.items():
            for s, mean, se in zip(subsidies, stats["mean"], stats["se"]):
                writer.writerow([dev, s, mean, se])


def load_results_from_csv(folder):
    obj_dict = defaultdict(lambda: {"mean": [], "se": []})
    h2_dict = defaultdict(lambda: {"mean": [], "se": []})

    obj_file = os.path.join(folder, "objective_results.csv")
    h2_file = os.path.join(folder, "hydrogen_results.csv")

    with open(obj_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dev = float(row["deviation"])
            obj_dict[dev]["mean"].append(float(row["mean"]))
            obj_dict[dev]["se"].append(float(row["se"]))

    with open(h2_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dev = float(row["deviation"])
            h2_dict[dev]["mean"].append(float(row["mean"]))
            h2_dict[dev]["se"].append(float(row["se"]))

    return dict(obj_dict), dict(h2_dict)


# =========================
# Plotting
# =========================
def plot_objective_values(objective_dict, value_range, folder):
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(10, 5))

    for label, stats in objective_dict.items():
        plt.errorbar(
            value_range,
            stats["mean"],
            yerr=stats["se"],
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


def plot_hydrogen_production(h2_dict, value_range, folder):
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(10, 5))

    for label, stats in h2_dict.items():
        plt.errorbar(
            value_range,
            stats["mean"],
            yerr=stats["se"],
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


# =========================
# Experiment logic
# =========================
def subsidy_per_mwh_to_mscm(mwh_subsidies, gcv_mwh_per_kscm=2.78):
    return [s * gcv_mwh_per_kscm * 1000 for s in mwh_subsidies]


def subsidy_experiment(G, subsidies_mwh, amount_per_point, deviation=0):
    subsidies = subsidy_per_mwh_to_mscm(subsidies_mwh)

    G_changed = change_hydrogen_price(
        G,
        subsidies,
        amount_per_point,
        variable_name="generation_cost"
    )

    s_list = create_scenario_trees(G_changed, dev=deviation)

    return results_from_subsidy(
        G, s_list, subsidies_mwh, False, args.time_limit, deviation=deviation
    )


def sub_dev_experiment(G, subsidies, deviations, amount_per_point):
    obj_dict = {}
    h2_dict = {}

    for dev in deviations:
        print("deviation", dev)

        obj_means, obj_se, h2_means, h2_se = subsidy_experiment(
            G, subsidies, amount_per_point, deviation=dev
        )

        obj_dict[dev] = {"mean": obj_means, "se": obj_se}
        h2_dict[dev] = {"mean": h2_means, "se": h2_se}

    return obj_dict, h2_dict


# =========================
# Main
# =========================
if __name__ == "__main__":

    if args.replot_only:
        print("Replotting from CSV...")

        obj_dict, h2_dict = load_results_from_csv(FIGURES_FOLDER)

    else:
        print("Running full experiment...")

        G = scpf.build_base_graph()

        obj_dict, h2_dict = sub_dev_experiment(
            G,
            args.subsidies,
            args.deviations,
            args.amount_per_point
        )

        save_results_to_csv(
            obj_dict,
            h2_dict,
            args.subsidies,
            FIGURES_FOLDER
        )

    # Always plot (from loaded or computed data)
    plot_objective_values(obj_dict, args.subsidies, FIGURES_FOLDER)
    plot_hydrogen_production(h2_dict, args.subsidies, FIGURES_FOLDER)