import os, sys
import time
import matplotlib.pyplot as plt

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import study_case_stochastic_model as scsm
import study_case_problem_file as scsp

# -----------------------------
# Settings
# -----------------------------
NETWORKS = {
    "smaller_network": "data/data_analysis_results/Geojson_pipelines/smaller_network.geojson",
    "bigger_network": "data/data_analysis_results/Geojson_pipelines/bigger_network.geojson",
    "study_case_network": "data/data_analysis_results/Geojson_pipelines/study_case_network.geojson",
}

DATA_FOLDERS = {
    "bigger_network": "study_case_model/compare_models/bigger_network/scenario_variables/",
    "smaller_network": "study_case_model/compare_models/smaller_network/scenario_variables/",
    "study_case_network": "study_case_model/compare_models/study_case_network/scenario_variables/",
}

NUMBER_OF_STAGES = 3
# We'll vary stage 2 and stage 3 branching
STAGE_2_OPTIONS = [3, 5, 7]
STAGE_3_OPTIONS = [1, 2, 4]


# -----------------------------
# Helper function
# -----------------------------
def run_model(geojson_file, data_folder, branches_per_stage):
    # Build graph
    G = scsp.build_base_graph(geojson_file)
    # Create scenarios
    scenarios = scsp.create_scenarios(NUMBER_OF_STAGES, branches_per_stage, G, data_folder)
    # Build model
    model = scsm.create_model(G, scenarios)
    
    # Solve and time
    start_time = time.time()
    results = scsm.solve_model(model)
    solve_time = time.time() - start_time
    
    return solve_time, results

# -----------------------------
# Run experiments
# -----------------------------
results_summary = []

for net_name, geojson_file in NETWORKS.items():
    data_folder = DATA_FOLDERS[net_name]
    
    print(f"\n=== Network: {net_name} ===")
    
    for stage2 in STAGE_2_OPTIONS:
        for stage3 in STAGE_3_OPTIONS:
            branches_per_stage = {1: 1, 2: stage2, 3: stage3}
            
            print(f"Running with stage2={stage2}, stage3={stage3}...")
            solve_time, results = run_model(geojson_file, data_folder, branches_per_stage)
            
            results_summary.append({
                "network": net_name,
                "stage2": stage2,
                "stage3": stage3,
                "solve_time": solve_time,
                "results": results
            })
            print(f" --> Solve time: {solve_time:.2f}s")

# -----------------------------
# Plot solve times
# -----------------------------
plt.figure(figsize=(10, 6))
for net_name in NETWORKS.keys():
    times = [r["solve_time"] for r in results_summary if r["network"] == net_name]
    labels = [f"{r['stage2']}-{r['stage3']}" for r in results_summary if r["network"] == net_name]
    plt.plot(labels, times, marker='o', label=net_name)

plt.xlabel("Stage2-Stage3 branching")
plt.ylabel("Solve Time (s)")
plt.title("Comparison of Solve Time for Different Networks and Scenario Branching")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()