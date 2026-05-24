import matplotlib.pyplot as plt
from pyomo.opt import TerminationCondition
import numpy as np
import pyomo.environ as pyo
from pathlib import Path
import pandas as pd
import time
import os
import sys
import argparse

# Add the parent directory to the Python path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from study_case_stochastic_model import solve_model, create_model, save_model_values
from study_case_problem_file import build_base_graph,  create_scenarios
from experiment_utils import subsidy_per_mwh_to_mscm, apply_subsidy

# =========================
# Argument parser
# =========================
parser = argparse.ArgumentParser(description="Solve times experiments")

# Folder
parser.add_argument(
    "--folder",
    type=str,
    default="study_case_model/figures/solve_times/",
    help="Folder to save figures and CSVs"
)

parser.add_argument(
    "--pickle_folder",
    type=str,
    default="study_case_model/figures/solve_times/",
    help="Folder to save Pickle folder"
)

parser.add_argument(
    "--threads",
    type = str,
    default=8,
    help = "Amount of threads for the model to run on"
)

parser.add_argument(
    "--branches_stage2",
    type=int,
    default=4,
    help="Number of branches in stage 2"
)
parser.add_argument(
    "--branches_stage3",
    type=int,
    default=4,
    help="Number of branches in stage 3"
)

parser.add_argument(
    "--precision",
    type=float,
    default=0.002,
    help="Solver precision / tolerance"
)

parser.add_argument(
    "--subsidy",
    type= float,
    default=0.0,
    help = "subsidy level"
)

parser.add_argument(
    "--deviation",
    type= float,
    default =0.0,
    help = 'Allowed deviation'
)

parser.add_argument(
    "--homogeneous_splits",
    type = int,
    default=10,
    help = "Number of homogeneous splits per split arc"
    )

parser.add_argument(
    "--density_bounds",
    type = int,
    default=1,
    help = "Number of density bounds"
)

parser.add_argument(
    "--runs",
    type= int,
    default=2,
    help = "Runs per experiment"
)

# =========================
# Parse arguments
# =========================
args = parser.parse_args()

# =========================
# Map to variables
# =========================
FOLDER = args.folder
NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {1: 1, 2: args.branches_stage2, 3: args.branches_stage3}
PRECISION = args.precision
RUNS= args.runs
DENSITY_BOUNDS = args.density_bounds
THREADS = args.threads
DEVIATION = args.deviation
SUBSIDY = args.subsidy
HOMOGENEOUS_SPLITS = args.homogeneous_splits


def time_model(model, precision = PRECISION, threads = THREADS):
    start_time = time.time()
    results = solve_model(model, verbose=False, precision = precision, threads = threads)
    end_time = time.time()
    return end_time - start_time

def add_row_dict_to_csv(row, folder, filename):
    Path(folder).mkdir(parents=True, exist_ok=True)

    file_path = os.path.join(folder, filename)

    # Convert row dict to single-row DataFrame
    new_row_df = pd.DataFrame([row])

    # Append if file exists, otherwise create new file
    if os.path.exists(file_path):
        new_row_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        new_row_df.to_csv(file_path, mode='w', header=True, index=False)

    print(f"Saved to {file_path}")

def create_row(model, solve_time, run):
    row = {
        "subsidy": args.subsidy,
        "deviation": args.deviation,
        "run": run, 
        "solve_time": solve_time,
    }
    h2_vals = [pyo.value(model.h2_production[m_3]) for m_3 in model.M[3]]
    row["avg_h2_production"] = np.mean(h2_vals)
    row["objective_value"] = pyo.value(model.objective)

    return row


if __name__ == "__main__":
    G = build_base_graph()

    # Convert subsidy
    subsidy_mscm = subsidy_per_mwh_to_mscm(SUBSIDY)

    # Apply subsidy
    G = apply_subsidy(G, subsidy_mscm)


    for RUN in range(0,RUNS):
        print("Solving model:" + f" dev{DEVIATION}, sub{SUBSIDY}, run{RUN}", flush=True)
        scenarios = create_scenarios(NUMBER_OF_STAGES, BRANCHES_PER_STAGE, G, seed=RUN)
        model = create_model(G, scenarios, allowed_deviation=DEVIATION, number_of_density_bounds=DENSITY_BOUNDS, splits_per_arc=np.linspace(0, 1, HOMOGENEOUS_SPLITS))
        time_taken = time_model(model)
        row = create_row(model, time_taken, RUN)
        add_row_dict_to_csv(row, FOLDER + f"sub{SUBSIDY}/dev{DEVIATION}/run{RUN}/",  "solve_times.csv")
        save_model_values(model, args.pickle_folder +  f"sub{SUBSIDY}/dev{DEVIATION}/run{RUN}/model.pkl")
    print("== Done ==")