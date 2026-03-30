import os, sys
import pickle
from pathlib import Path
# Add the parent directory to the Python path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import study_case_stochastic_model as scsm
import study_case_problem_file as scsp

DATA_FOLDER = "study_case_model/unit_networks/scenario_variables/"
FOLDER = "study_case_model/figures/unit_networks/"
PICKLE_FOLDER = "study_case_model/figures/unit_networks/"
NETWORK_FOLDER = "study_case_model/unit_networks_solves/unit_gen2_networks/"

NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {1 : 1, 2 : 1, 3: 1}

def load_unit_network(filename):
    """
    Load a unit network graph from a pickle file.

    Parameters:
        filename (str): Path to the pickle file

    Returns:
        nx.DiGraph: The loaded graph
    """
    with open(filename, "rb") as f:
        G = pickle.load(f)
    return G


def solve_small_network(path, file):
    print("=======")
    print(f"Solving network: {file}")
    print("=======")

    # Load network
    G = load_unit_network(path)

    # Create scenarios
    scenarios = scsp.create_scenarios(
        NUMBER_OF_STAGES,
        BRANCHES_PER_STAGE,
        G,
        folder= DATA_FOLDER
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
    print_objectives()