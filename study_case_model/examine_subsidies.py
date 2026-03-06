import study_case_problem_file as scpf
import numpy as np
import study_case_stochastic_model as scsm
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import os

NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {1 : 1, 2 : 3, 3: 3}
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

def create_scenario_trees(G_list):
    scenario_list = []
    for i, G_part in enumerate(G_list):
        s_part =[]
        for j, G in enumerate(G_part):
            s_part.append(scpf.create_scenarios(NUMBER_OF_STAGES, BRANCHES_PER_STAGE, G, folder=DATA_FOLDER + f"demand/{i}/{j}/"))
        scenario_list.append(s_part)
    return scenario_list

def solve_multiple_problems(G, s_list, verbose = False, time_limit= None):
    result_list = []
    avg_hydrogen_production = []
    for scenarios in s_list:
            model = scsm.create_model(G, scenarios)
            _ = scsm.solve_model(model, verbose, time_limit)
            result_list.append(pyo.value(model.objective))
            values = [pyo.value(model.h2_production[i]) for i in model.h2_production]
            avg_hydrogen_production.append(np.mean(values))
    return result_list, avg_hydrogen_production

def results_from_subsidy(G, s_super_list, verbose = False, time_limit =None):
    objective_averages = []
    objective_variance = []
    hydrogen_production = []
    for s_part in s_super_list:
        results, avg_hydrogen_production = solve_multiple_problems(G, s_part, verbose, time_limit)
        objective_averages.append(np.mean(results))
        objective_variance.append(np.var(results))
        hydrogen_production.append(np.mean(avg_hydrogen_production))

    return objective_averages, objective_variance, hydrogen_production

def plot_objective_values(objective_averages, value_range, folder, variable = "unknown"):
    os.makedirs(folder, exist_ok=True)
    # Plot 1: Objective averages
    plt.figure(figsize=(10, 5))
    plt.plot(value_range, objective_averages, 'o-')
    plt.xlabel(f'Subsidy (Euro/Mwh)')
    plt.ylabel('Objective Average')
    plt.title(f'Objective Average vs Subsidy')
    plt.grid(True)
    plt.savefig(folder+ f'Objective Average vs Subsidy')
    plt.show()

def plot_objective_variance(variance, value_range, folder, variable = "unknown"):
    os.makedirs(folder, exist_ok=True)
    # Plot 2: Objective variance
    plt.figure(figsize=(10, 5))
    plt.plot(value_range, variance, 's-', color='orange')
    plt.xlabel(f'Subsidy (Euro/Mwh)')
    plt.ylabel('Objective Variance')
    plt.title(f'Objective Variance vs {variable} standard deviation')
    plt.grid(True)
    plt.savefig(folder+ f"changing_{variable}_objective_variance")
    plt.show()


def plot_hydrogen_production(h2_production, value_range, folder, variable="unknown"):
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(
        value_range,
        h2_production,
        'o-',
        color='purple'
    )
    plt.xlabel(f'Subsidy (Euro/Mwh)')
    plt.ylabel('Hydrogen Production')
    plt.title(f'Subsidy and Hydrogen production')
    plt.grid(True)
    plt.savefig(folder + f"hydrogen_production")
    plt.show()

def subsidy_per_mwh_to_mscm(mwh_subsidies, gcv_mwh_per_kscm=2.78):
    mwh_per_mscm = gcv_mwh_per_kscm * 1000  # 3350 MWh per MSCM
    return  [s * mwh_per_mscm for s in mwh_subsidies]



if __name__ == "__main__":
    G = scpf.build_base_graph()

    amount_per_point = 2
    subsidies_mwh = [0, 20, 40, 60, 80] # Euro per MWh
    subsidies = subsidy_per_mwh_to_mscm(subsidies_mwh)

    G_changed_hydrogen_cost = change_hydrogen_price(G, subsidies, amount_per_point, variable_name= "generation_cost")
    s_list = create_scenario_trees(G_changed_hydrogen_cost)
    objective_averages, variance, h2_production = results_from_subsidy(G, s_list, False, 40)
    plot_objective_values(objective_averages, subsidies_mwh, folder = FIGURES_FOLDER, variable="Demand")
    plot_objective_variance(variance,  subsidies_mwh, folder = FIGURES_FOLDER,  variable="Demand")
    plot_hydrogen_production(h2_production, subsidies_mwh, folder=FIGURES_FOLDER, variable="Price Long Term")
