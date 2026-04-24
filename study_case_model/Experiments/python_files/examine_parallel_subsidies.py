import numpy as np
import pyomo.environ as pyo
import pandas as pd
import sys
import os
import argparse
from pathlib import Path

# Add parent directory
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import study_case_stochastic_model as scsm
import study_case_problem_file as scpf
from Experiments.python_files.experiment_utils import subsidy_per_mwh_to_mscm, apply_subsidy


# =========================
# Argument parsing
# =========================
parser = argparse.ArgumentParser()

parser.add_argument("--run", type=int, default= 0 )
parser.add_argument("--branches_stage2", type=int, default=8)
parser.add_argument("--branches_stage3", type=int, default=8)
parser.add_argument("--subsidy", type=float, default=0)
parser.add_argument("--deviation", type=float, default=0)

parser.add_argument("--upper_bounds", type=int, default=1)
parser.add_argument("--time_limit", type=float, default=None)
parser.add_argument("--threads", type= int, default= 8)
parser.add_argument("--precision", type=float, default=0.001)

parser.add_argument("--data_folder", type=str, default="scenario_variables/other_experiment_data/")
parser.add_argument("--pickle_folder", type= str, default= "study_case_model/figures/other_experiments")

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
SUBSIDY = args.subsidy
DEVIATION = args.deviation
RUN = args.run


# =========================
# Main execution
# =========================
if __name__ == "__main__":

    # Build base graph
    G = scpf.build_base_graph()

    # Convert subsidy
    subsidy_mscm = subsidy_per_mwh_to_mscm(SUBSIDY)

    # Apply subsidy
    G_changed = apply_subsidy(G, subsidy_mscm)

    # Create scenario folder
    data_folder = os.path.join(
        args.data_folder,
        f"dev{DEVIATION}",
        f"sub{SUBSIDY}",
        f"run{RUN}"
    )

    # Create pickle folder
    pickle_folder = os.path.join(
        args.pickle_folder,
        f"dev{DEVIATION}",
        f"sub{SUBSIDY}",
        f"run{RUN}"
    )

    os.makedirs(data_folder, exist_ok=True)

    # Create scenarios
    scenarios = scpf.create_scenarios(
        NUMBER_OF_STAGES,
        BRANCHES_PER_STAGE,
        G_changed,
        seed= RUN,
        folder=data_folder
    )

    print("Solving model:" + f" dev{DEVIATION}, sub{SUBSIDY}, run{RUN}", flush=True)

    # Build model
    model = scsm.create_model(
        G_changed,
        scenarios,
        allowed_deviation=DEVIATION,
        number_of_density_bounds=UPPER_BOUNDS
    )

    # Solve with Gurobi (multithreaded)
    node_file_folder = os.environ.get("TMPDIR", "/tmp")
    node_file_folder = os.path.join(node_file_folder, f"gurobi_dev{DEVIATION}_sub{SUBSIDY}_run{RUN}")
    results = scsm.solve_model(model, threads= THREADS, verbose= True, precision=args.precision, node_file_folder=node_file_folder)

    # Save results
    scsm.save_model_values(model, os.path.join(pickle_folder, "model_snapshot.pkl"))

    print("Finished successfully!", flush=True)