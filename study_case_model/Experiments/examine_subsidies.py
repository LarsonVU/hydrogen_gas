import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import study_case_stochastic_model as scsm
import study_case_problem_file as scpf

NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {1 : 1, 2 : 5, 3: 4}
DATA_FOLDER = "study_case_model/scenario_variables/examine_subsidies/"
FIGURES_FOLDER = "study_case_model/figures/examine_subsidies/"

def change_hydrogen_price(G, value_range = [0, 30, 70], amount_per_point = 5, variable_name = "unknown"):
    G_list = []
    for var in value_range:
        G_list_part = []
        for i in range(amount_per_point):
            G_changed = G.copy()
            for node in G.nodes:
                if G.nodes[node][variable_name] is not None:

                    if  G.nodes[node]["component_ratio"]["H2"] >0:
                        G_changed.nodes[node][variable_name] = float(G.nodes[node][variable_name]) - var
            G_list_part.append(G_changed)
        G_list.append(G_list_part)
    return G_list

def create_scenario_trees(G_list, dev):
    scenario_list = []
    for i, G_part in enumerate(G_list):
        s_part =[]
        for j, G in enumerate(G_part):
            s_part.append(scpf.create_scenarios(NUMBER_OF_STAGES, BRANCHES_PER_STAGE, G, folder=DATA_FOLDER + f"deviation{dev}/{i}/{j}"))
        scenario_list.append(s_part)
    return scenario_list

def solve_multiple_problems(G, s_list, i, verbose=False, time_limit=None, allowed_deviation = 0):
    result_list = []
    avg_hydrogen_production = []

    for j, scenarios in enumerate(s_list):
        print("run " ,j)
        model = scsm.create_model(G, scenarios, allowed_deviation=allowed_deviation)
        _ = scsm.solve_model(model, verbose, time_limit, precision=0.01)

        # --- Save solved model ---
        folder = DATA_FOLDER + f"deviation{allowed_deviation}/subsidy{i}/run{j}/"
        filename = folder + "model_snapshot.pkl"
        scsm.save_model_values(model, filename)

        result_list.append(pyo.value(model.objective))

        values = [pyo.value(model.h2_production[i]) for i in model.h2_production]
        avg_hydrogen_production.append(np.mean(values))

    return result_list, avg_hydrogen_production

def results_from_subsidy(G, s_super_list, subsidies, verbose=False, time_limit=None, deviation =0):

    objective_means = []
    objective_se = []

    hydrogen_means = []
    hydrogen_se = []

    for s, s_part in zip(subsidies, s_super_list):
        print("subsidy ", s)
        results, avg_hydrogen_production = solve_multiple_problems(
            G, s_part, s, verbose, time_limit, allowed_deviation=deviation
        )

        n = len(results)

        objective_means.append(np.mean(results))
        objective_se.append(np.std(results) / np.sqrt(n))

        hydrogen_means.append(np.mean(avg_hydrogen_production))
        hydrogen_se.append(np.std(avg_hydrogen_production) / np.sqrt(n))

    return objective_means, objective_se, hydrogen_means, hydrogen_se

def plot_objective_values(objective_dict, value_range, folder):
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(10, 5))

    for label, stats in objective_dict.items():
        means = stats["mean"]
        se = stats["se"]

        plt.errorbar(
            value_range,
            means,
            yerr=se,
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
    plt.show()

def plot_hydrogen_production(h2_dict, value_range, folder):
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(10, 5))

    for label, stats in h2_dict.items():
        means = stats["mean"]
        se = stats["se"]

        plt.errorbar(
            value_range,
            means,
            yerr=se,
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
    plt.show()


def subsidy_per_mwh_to_mscm(mwh_subsidies, gcv_mwh_per_kscm=2.78):
    mwh_per_mscm = gcv_mwh_per_kscm * 1000  # 3350 MWh per MSCM
    return  [s * mwh_per_mscm for s in mwh_subsidies]


def subsidy_experiment(G, subsidies_mwh, deviation = 0):
    G_changed_hydrogen_cost = change_hydrogen_price(G, subsidies, amount_per_point, variable_name= "generation_cost")
    s_list = create_scenario_trees(G_changed_hydrogen_cost, dev=deviation)

    objective_means, objective_se, h2_means, h2_se = results_from_subsidy(
        G, s_list, subsidies_mwh, False, deviation=deviation
    )
    return objective_means, objective_se, h2_means, h2_se

def sub_dev_experiment(G, subsidies, deviations):
    obj_dict = {}
    h2_dict = {}

    for dev in deviations:
        print("deviation", dev)
        objective_means, objective_se, h2_means, h2_se = subsidy_experiment(
            G, subsidies, deviation=dev
        )

        obj_dict[dev] = {
            "mean": objective_means,
            "se": objective_se
        }

        h2_dict[dev] = {
            "mean": h2_means,
            "se": h2_se
        }

    return obj_dict, h2_dict

if __name__ == "__main__":
    G = scpf.build_base_graph()

    amount_per_point = 2
    subsidies_mwh = [40 * i for i in range(3)] # Euro per MWh
    subsidies = subsidy_per_mwh_to_mscm(subsidies_mwh)

    # objective_means, objective_se, h2_means, h2_se = subsidy_experiment(G, subsidies)

    # obj_dict = {"0%" : {"mean" : objective_means, "se" : objective_se}}
    # h2_dict = {"0%" : {"mean" : h2_means, "se" : h2_se}}

    obj_dict, h2_dict = sub_dev_experiment(G, subsidies_mwh, [0, 0.05, 0.1])


    plot_objective_values(
        obj_dict,
        subsidies_mwh,
        folder=FIGURES_FOLDER
    )

    plot_hydrogen_production(
        h2_dict,
        subsidies_mwh,
        folder=FIGURES_FOLDER
    )



    

