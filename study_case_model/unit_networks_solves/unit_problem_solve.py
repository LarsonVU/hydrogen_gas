import os, sys
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import study_case_stochastic_model as scsm
import study_case_problem_file as scsp
import mini_network_maker as mwm

DATA_FOLDER = "study_case_model/smaller_network/scenario_variables/"
FOLDER = "study_case_model/unit_networks_solves/figures/"
PICKLE_FOLDER = "study_case_model/unit_networks_solves/snapshots/"
NETWORK_FOLDER = "study_case_model/unit_networks"

NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {1 : 1, 2 : 1, 3: 1}

def solve_small_network(path, file):
    print("=======")
    print(f"Solving network: {file}")
    print("=======")

    # Load network
    G = mwm.load_unit_network(path)

    # Create scenarios
    scenarios = scsp.create_scenarios(
        NUMBER_OF_STAGES,
        BRANCHES_PER_STAGE,
        G,
        DATA_FOLDER
    )

    # Build model
    model = scsm.create_model(G, scenarios)

    # Solve model
    results = scsm.solve_model(model, time_limit=100)

    print(results)

    # Plot results
    scsm.plot_results(model, folder=FOLDER + file + "/")
    scsm.save_model_values(model, filename = PICKLE_FOLDER + file)
    return

def solve_small_networks():

    for file in os.listdir(NETWORK_FOLDER):

        if file.endswith(".pkl"):   # only load stored networks
            path = os.path.join(NETWORK_FOLDER, file)
            solve_small_network(path, file)
    return

def print_objectives():
    for file in os.listdir(PICKLE_FOLDER):
        if file.endswith(".pkl"):   # only load stored solutions
            path = os.path.join(PICKLE_FOLDER, file)
            model_values = scsm.load_param_values(path)
            print(file)
            print(model_values["objectives"])

if __name__ == "__main__":
    solve_small_networks()
    #print_objectives()

    #solve_small_network(os.path.join(NETWORK_FOLDER, "002_Ivysaur.pkl"), "002_Ivysaur.pkl")
    # model_values = scsm.load_param_values(PICKLE_FOLDER + "027_Sandshrew.pkl")
    # print("fuel")
    # scsm.print_select_model_values(value_dict = model_values["variables"]["w"])
    # print("flow")
    # scsm.print_select_model_values(value_dict = model_values["variables"]["f"])
    # print("booking (entry)")
    # scsm.print_select_model_values(value_dict = model_values["variables"]["x_entry"])
    # print("booking (exit)")
    # scsm.print_select_model_values(value_dict = model_values["variables"]["x_exit"])
    # print("pressure (into pipe)")
    # scsm.print_select_model_values(value_dict = model_values["variables"]["p_in"])
    # print("pressure (out of pipe)")
    # scsm.print_select_model_values(value_dict = model_values["variables"]["p_out"])
