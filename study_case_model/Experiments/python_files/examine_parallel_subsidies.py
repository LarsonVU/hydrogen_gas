import numpy as np
import pyomo.environ as pyo
import pandas as pd
import sys
import os
import argparse

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import study_case_stochastic_model as scsm
import study_case_problem_file as scpf


# =========================
# Argument parsing
# =========================
parser = argparse.ArgumentParser()

parser.add_argument("--amount_per_point", type=int, default=2)
parser.add_argument("--branches_stage2", type=int, default=2)
parser.add_argument("--branches_stage3", type=int, default=2)
parser.add_argument("--subsidies", type=float, nargs="+", default=[0, 40, 80])
parser.add_argument("--deviations", type=float, nargs="+", default=[0])
parser.add_argument("--upper_bounds", type=int, default=1)
parser.add_argument("--time_limit", type=float, default=None)
parser.add_argument("--threads", type= int, default= 1)

parser.add_argument("--data_folder", type=str, required=True)

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
THREADS = args.threads

# =========================
# Helper functions
# =========================
def subsidy_per_mwh_to_mscm(mwh_subsidy, gcv_mwh_per_kscm=2.78):
    return mwh_subsidy * gcv_mwh_per_kscm * 1000


def apply_subsidy(G, subsidy_value, variable_name="generation_cost"):
    G_changed = G.copy()

    for node in G.nodes:
        if not pd.isna(G.nodes[node][variable_name]):
            if G.nodes[node]["component_ratio"]["H2"] > 0:
                G_changed.nodes[node][variable_name] = (
                    float(G.nodes[node][variable_name]) - subsidy_value
                )
    return G_changed

# =========================
# SLURM ARRAY INDEXING
# =========================
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

n_sub = len(args.subsidies)
n_dev = len(args.deviations)
n_runs = args.amount_per_point

total = n_sub * n_dev * n_runs

if task_id >= total:
    raise ValueError(f"Task ID {task_id} exceeds total jobs {total}")

# Decode index
dev_idx = task_id // (n_sub * n_runs)
rem = task_id % (n_sub * n_runs)

sub_idx = rem // n_runs
run_idx = rem % n_runs

deviation = args.deviations[dev_idx]
subsidy = args.subsidies[sub_idx]

print(f"[TASK {task_id}] deviation={deviation}, subsidy={subsidy}, run={run_idx}", flush=True)


# =========================
# Main execution
# =========================
if __name__ == "__main__":

    # Build base graph
    G = scpf.build_base_graph()

    # Convert subsidy
    subsidy_mscm = subsidy_per_mwh_to_mscm(subsidy)

    # Apply subsidy
    G_changed = apply_subsidy(G, subsidy_mscm)

    # Create scenario folder
    folder = os.path.join(
        args.data_folder,
        f"dev{deviation}",
        f"sub{subsidy}",
        f"run{run_idx}"
    )
    os.makedirs(folder, exist_ok=True)

    # Create scenarios
    scenarios = scpf.create_scenarios(
        NUMBER_OF_STAGES,
        BRANCHES_PER_STAGE,
        G_changed,
        folder=folder
    )

    print("Solving model...", flush=True)

    # Build model
    model = scsm.create_model(
        G_changed,
        scenarios,
        allowed_deviation=deviation,
        number_of_density_bounds=UPPER_BOUNDS
    )

    # Solve with Gurobi (multithreaded)
    results = scsm.solve_model(model, threads= THREADS, verbose= False, precision=0.001)

    # Save results
    scsm.save_model_values(model, os.path.join(folder, "model_snapshot.pkl"))

    print("Finished successfully!", flush=True)