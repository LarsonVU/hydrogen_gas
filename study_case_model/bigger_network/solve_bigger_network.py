import os, sys
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import study_case_stochastic_model as scsm
import study_case_problem_file as scsp

DATA_FOLDER = "study_case_model/bigger_network/scenario_variables/"
FOLDER = "study_case_model/bigger_network/figures/"
GEOJSON_FILE = "data/data_analysis_results/Geojson_pipelines/bigger_network.geojson"

NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {1 : 1, 2 : 5, 3: 2}

if __name__ == "__main__":
    G = scsp.build_base_graph(GEOJSON_FILE)
    scenarios = scsp.create_scenarios(NUMBER_OF_STAGES, BRANCHES_PER_STAGE, G, DATA_FOLDER)

    model = scsm.create_model(G, scenarios)

    results = scsm.solve_model(model, time_limit=100)
    print(results)
    scsm.plot_results(model, folder = FOLDER)