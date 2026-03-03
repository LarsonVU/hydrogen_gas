import study_case_problem_file as scpf
import numpy as np
import study_case_stochastic_model as scsm
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import os

NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {1 : 1, 2 : 3, 3: 3}
DATA_FOLDER = "study_case_model/scenario_variables/changed_var/"
FIGURES_FOLDER = "study_case_model/figures/changed_demand/"

def change_var(G, value_range = (0,0.5), amount_of_points =4, amount_per_point = 5, variable_name = "unknown"):
    G_list = []
    for var in np.linspace(value_range[0], value_range[1], amount_of_points):
        G_list_part = []
        for i in range(amount_per_point):
            G_changed = G.copy()
            for node in G.nodes:
                if G.nodes[node][variable_name] is not None:
                    G.nodes[node][variable_name] = var
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
    scenario_variances= []
    for scenarios in s_list:
            model = scsm.create_model(G, scenarios)
            _ = scsm.solve_model(model, verbose, time_limit)
            result_list.append(pyo.value(model.objective))
            scenario_variances.append(pyo.value(model.scenario_objective_variance))
    return result_list, scenario_variances

def results_from_variance(G, s_super_list, verbose = False, time_limit =None):
    objective_averages = []
    objective_variance = []
    scenario_variance = []
    for s_part in s_super_list:
        results, variances = solve_multiple_problems(G, s_part, verbose, time_limit)
        objective_averages.append(np.mean(results))
        objective_variance.append(np.var(results))
        scenario_variance.append(np.mean(variances))

    return objective_averages, objective_variance, scenario_variance

def plot_objective_values(var_range, amount_of_points, folder, variable = "unknown"):
    os.makedirs(folder, exist_ok=True)
    # Plot 1: Objective averages
    plt.figure(figsize=(10, 5))
    plt.plot(np.linspace(var_range[0], var_range[1], amount_of_points), objective_averages, 'o-')
    plt.xlabel(f'{variable} standard deviation')
    plt.ylabel('Objective Average')
    plt.title(f'Objective Average vs {variable} standard deviation')
    plt.grid(True)
    plt.savefig(folder+ f"changing_{variable}_objective_values")
    plt.show()

def plot_objective_variance(var_range, amount_of_points, folder, variable = "unknown"):
    os.makedirs(folder, exist_ok=True)
    # Plot 2: Objective variance
    plt.figure(figsize=(10, 5))
    plt.plot(np.linspace(var_range[0], var_range[1], amount_of_points), variance, 's-', color='orange')
    plt.xlabel(f'{variable} standard deviation')
    plt.ylabel('Objective Variance')
    plt.title(f'Objective Variance vs {variable} standard deviation')
    plt.grid(True)
    plt.savefig(folder+ f"changing_{variable}_objective_variance")
    plt.show()


def plot_scenario_variance(var_range, amount_of_points, folder, variable="unknown"):
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(
        np.linspace(var_range[0], var_range[1], amount_of_points),
        scenario_variance,
        'o-',
        color='purple'
    )
    plt.xlabel(f'{variable} standard deviation')
    plt.ylabel('Scenario Variance')
    plt.title(f'Scenario Variance vs {variable} Variance')
    plt.grid(True)
    plt.savefig(folder + f"changing_{variable}_scenario_variance")
    plt.show()


if __name__ == "__main__":
    G = scpf.build_base_graph()

    var_range = (0,0.5)
    amount_of_points = 2
    amount_per_point = 2

    G_changed_var_demand = change_var(G, var_range, amount_of_points, amount_per_point, variable_name= "demand_variance")
    s_list = create_scenario_trees(G_changed_var_demand)
    objective_averages, variance, scenario_variance = results_from_variance(G, s_list, False, 20)
    plot_objective_values(var_range, amount_of_points, folder = FIGURES_FOLDER, variable="Demand")
    plot_objective_variance(var_range, amount_of_points, folder = FIGURES_FOLDER,  variable="Demand")
    plot_scenario_variance(var_range, amount_of_points, folder=FIGURES_FOLDER, variable="Price Long Term")

    var_range = (0,0.5)
    amount_of_points = 6
    amount_per_point = 5

    G_changed_var_price = change_var(G, var_range, amount_of_points, amount_per_point, variable_name= "long_term_price_std")
    s_list = create_scenario_trees(G_changed_var_price)
    objective_averages, variance, scenario_variance = results_from_variance(G, s_list, False, 20)
    plot_objective_values(var_range, amount_of_points, folder = FIGURES_FOLDER, variable="Price Long Term")
    plot_objective_variance(var_range, amount_of_points, folder = FIGURES_FOLDER,  variable="Price Long Term")
    plot_scenario_variance(var_range, amount_of_points, folder=FIGURES_FOLDER, variable="Price Long Term")  

    var_range = (0, 0.1)
    amount_of_points = 6
    amount_per_point = 5

    G_changed_var_price = change_var(G, var_range, amount_of_points, amount_per_point, variable_name= "day_ahead_price_std")
    s_list = create_scenario_trees(G_changed_var_price)
    objective_averages, variance, scenario_variance = results_from_variance(G, s_list, False, 20)
    plot_objective_values(var_range, amount_of_points, folder = FIGURES_FOLDER, variable="Price Day Ahead")
    plot_objective_variance(var_range, amount_of_points, folder = FIGURES_FOLDER,  variable="Price Day Ahead")
    plot_scenario_variance(var_range, amount_of_points, folder=FIGURES_FOLDER, variable="Price Day Ahead")  





    