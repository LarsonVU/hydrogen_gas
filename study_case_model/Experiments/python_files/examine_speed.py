
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

from study_case_stochastic_model import solve_model, create_model, generate_cutting_plane_pairs
from study_case_problem_file import build_base_graph,  create_scenarios

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
    default=0.001,
    help="Solver precision / tolerance"
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

def time_model(model, precision = PRECISION):
    start_time = time.time()
    results = solve_model(model, verbose=False, precision = precision)
    end_time = time.time()
    return end_time - start_time



def branch_solve_time_matrix(network, max_2, max_3, runs=4):
    G = network.copy()
    solve_times = {}

    for m_2 in range(2, max_2+1, 2):
        for m_3 in range(2, max_3+1, 2):

            times = []

            for r in range(runs):
                scenarios = create_scenarios(NUMBER_OF_STAGES, {1: 1, 2: m_2, 3: m_3}, G)
                model = create_model(G, scenarios, number_of_density_bounds=1)
                t = time_model(model)
                times.append(t)

            times = np.array(times)
            mean = np.mean(times)
            std = np.std(times, ddof=1)  # sample std
            stderr = std / np.sqrt(runs)

            solve_times[(m_2, m_3)] = (mean, stderr)

            print(
                f"Branches Stage 2: {m_2}, Stage 3: {m_3} | "
                f"Mean: {mean:.2f}s ± {stderr:.2f}"
            )

    return solve_times
    
def plot_solve_time_matrix(solve_times, folder=FOLDER):
    os.makedirs(folder, exist_ok=True)

    m_2_values = sorted(set(m_2 for m_2, m_3 in solve_times.keys()))
    m_3_values = sorted(set(m_3 for m_2, m_3 in solve_times.keys()))

    # Extract mean values for heatmap
    Z = np.array([
        [solve_times.get((m_2, m_3), (0, 0))[0] for m_3 in m_3_values]
        for m_2 in m_2_values
    ])

    plt.figure(figsize=(10, 6))
    plt.imshow(Z, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Mean Solve Time (seconds)')

    # Annotate with mean ± stderr
    for i, m_2 in enumerate(m_2_values):
        for j, m_3 in enumerate(m_3_values):
            mean, stderr = solve_times.get((m_2, m_3), (0, 0))
            plt.text(
                j, i,
                f'{mean:.2f}±{stderr:.2f}',
                ha='center', va='center',
                color='white',
                fontsize=9
            )

    plt.xticks(ticks=range(len(m_3_values)), labels=m_3_values)
    plt.yticks(ticks=range(len(m_2_values)), labels=m_2_values)

    plt.xlabel('Branches in Stage 3')
    plt.ylabel('Branches in Stage 2')
    plt.title(f'Solve Time Matrix (Tolerance = {PRECISION * 100}%)')

    plt.tight_layout()
    plt.savefig(folder + 'solve_time_matrix.png')
    plt.close()


def allowed_deviation_solve_times(network, scenarios, deviation_values):
    G = network.copy()
    solve_times = {}

    for deviation in deviation_values:
        model = create_model(G, scenarios, allowed_deviation=deviation, number_of_density_bounds=1)
        solve_times[deviation] = time_model(model)
        print(f"Allowed Deviation: {deviation}, Solve Time: {solve_times[deviation]} seconds")
    return solve_times

def plot_deviation_solve_times(solve_times):
    deviations = list(solve_times.keys())
    times = list(solve_times.values())

    plt.figure(figsize=(10, 6))
    plt.plot(deviations, times, marker='o')
    plt.xlabel('Allowed Deviation')
    plt.ylabel('Solve Time (seconds)')
    plt.title(f'Solve Time vs Allowed Deviation (Tolerance = {PRECISION *100}%)')
    plt.grid()
    plt.tight_layout()
    plt.savefig(FOLDER + 'deviation_solve_times.png')
    plt.close()

def cutting_planes_solve_times(network, scenarios, max_cutting_planes_in, max_cutting_planes_out):
    G = network.copy()
    cutting_plane_counts = {}

    for in_planes in range(0, max_cutting_planes_in + 1, 2):
        for out_planes in range(0, max_cutting_planes_out + 1, 2):
            cutting_plane_pairs = generate_cutting_plane_pairs(n_p_in=in_planes, n_p_out=out_planes)
            model = create_model(G, scenarios, cutting_plane_pairs=cutting_plane_pairs, number_of_density_bounds=1)
            solve_time = time_model(model)
            cutting_plane_counts[(in_planes, out_planes)] = solve_time
            print(f"Cutting Planes In: {in_planes}, Cutting Planes Out: {out_planes}, Solve Time: {solve_time} seconds")
    return cutting_plane_counts

def plot_cutting_plane_solve_times(cutting_plane_counts):
    in_planes = sorted(set(in_planes for in_planes, out_planes in cutting_plane_counts.keys()))
    out_planes = sorted(set(out_planes for in_planes, out_planes in cutting_plane_counts.keys()))

    Z = np.array([[cutting_plane_counts.get((in_planes, out_planes), 0) for out_planes in out_planes] for in_planes in in_planes])
    plt.figure(figsize=(10, 6))
    plt.imshow(Z, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Solve Time (seconds)')
    for i, in_plane in enumerate(in_planes):
        for j, out_plane in enumerate(out_planes):
            plt.text(j, i, f'{Z[i, j]:.2f}', ha='center', va='center', color='white', fontsize=10)
    plt.xticks(ticks=range(len(out_planes)), labels=out_planes)
    plt.yticks(ticks=range(len(in_planes)), labels=in_planes)
    plt.xlabel('Number of Cutting Planes Out')
    plt.ylabel('Number of Cutting Planes In')
    plt.title('Solve Time by Cutting Plane Counts')
    plt.tight_layout()
    plt.savefig(FOLDER + 'cutting_plane_solve_times.png')
    plt.close()

def density_solve_times(network, densities, runs=5):
    G = network.copy()
    
    # Store raw values
    raw_results = {d: {"times": []} for d in densities}

    for r in range(runs):
        print("run", r)

        # --- Common Random Numbers ---
        scenarios = create_scenarios(NUMBER_OF_STAGES, BRANCHES_PER_STAGE, G)

        for density in densities:
            model = create_model(
                G,
                scenarios,
                number_of_density_bounds=density
            )

            solve_time = time_model(model, precision=0.01)
            raw_results[density]["times"].append(solve_time)

    # --- Compute statistics ---
    results = {}

    for density in densities:
        times = raw_results[density]["times"]

        results[density] = {
            "time_mean": np.mean(times),
            "time_std": np.std(times),
        }

        print(
            f"Upperbounds: {density} | "
            f"Time: {results[density]['time_mean']:.3f} ± "
            f"{results[density]['time_std']/np.sqrt(runs):.3f} s"
        )

    return results

def plot_density_solve_times(results, precision=0.01, runs=5):
    densities = list(results.keys())
    time_means = [results[d]["time_mean"] for d in densities]
    time_err = [results[d]["time_std"] / np.sqrt(runs) for d in densities]

    plt.figure(figsize=(10, 6))
    plt.errorbar(densities, time_means, yerr=time_err, marker='o', capsize=5)
    plt.xlabel('Amount of upperbounds')
    plt.ylabel('Solve Time (seconds)')
    plt.title(f'Solve Time vs Amount of Upperbounds (Tolerance = {precision*100}%)')
    plt.grid()
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(FOLDER + 'upperbounds_solve_times.png')
    plt.close()

def splits_per_arc_experiment(network, splits_values, runs=5):
    G = network.copy()
    
    # Store raw values first
    raw_results = {s: {"times": [], "objectives": []} for s in splits_values}

    for r in range(runs):
        print("run ", r )
        # --- Common Random Numbers ---
        scenarios = create_scenarios(NUMBER_OF_STAGES, BRANCHES_PER_STAGE, G)

        for splits in splits_values:

            model = create_model(
                G,
                scenarios,
                splits_per_arc=np.linspace(0, 1, splits),
                number_of_density_bounds= 1
            )

            solve_time = time_model(model, precision=0.01)
            obj_value = pyo.value(model.objective)

            raw_results[splits]["times"].append(solve_time)
            raw_results[splits]["objectives"].append(obj_value)

    # --- Compute statistics ---
    results = {}

    for splits in splits_values:
        times = raw_results[splits]["times"]
        objectives = raw_results[splits]["objectives"]

        results[splits] = {
            "time_mean": np.mean(times),
            "time_std": np.std(times),
            "objective_mean": np.mean(objectives),
            "objective_std": np.std(objectives),
        }

        print(
            f"Splits per arc: {splits} | "
            f"Time: {results[splits]['time_mean']:.3f} ± {results[splits]['time_std']/np.sqrt(runs):.3f} s | "
            f"Objective: {results[splits]['objective_mean']:.3f} ± {results[splits]['objective_std']/np.sqrt(runs):.3f}"
        )

    return results

def plot_splits_solve_times(results, precision=0.01, runs =5):
    splits = list(results.keys())
    time_means = [results[s]["time_mean"] for s in splits]
    time_err = [results[s]["time_std"]/np.sqrt(runs) for s in splits]

    plt.figure(figsize=(10, 6))
    plt.errorbar(splits, time_means, yerr=time_err, marker='o', capsize=5)
    plt.xlabel('Splits per Arc')
    plt.ylabel('Solve Time (seconds)')
    plt.title(f'Solve Time vs Splits per Arc (Tolerance = {precision*100}%)')
    plt.grid()
    plt.ylim(bottom = 0)
    plt.tight_layout()
    plt.savefig(FOLDER + 'splits_per_arc_solve_times.png')
    plt.close()

def plot_splits_objective(results, precision=0.01, runs =5):
    splits = list(results.keys())
    obj_means = [results[s]["objective_mean"] for s in splits]
    obj_err = [results[s]["objective_std"]/np.sqrt(runs) for s in splits]

    plt.figure(figsize=(10, 6))
    plt.errorbar(splits, obj_means, yerr=obj_err, marker='o', capsize=5)
    plt.xlabel('Splits per Arc')
    plt.ylabel('Objective Value')
    plt.title(f'Objective Value vs Splits per Arc (Tolerance = {precision*100}%)')
    plt.grid()
    plt.ylim(bottom = 0)
    plt.tight_layout()
    plt.savefig(FOLDER + 'splits_per_arc_objective.png')
    plt.close()

def save_dict_to_csv(data_dict, folder, filename):
    Path(folder).mkdir(parents=True, exist_ok=True)

    # Convert dict to DataFrame
    df = pd.DataFrame.from_dict(data_dict, orient="index")

    # Case 1: tuple values (mean, stderr)
    if df.shape[1] == 2:
        df.columns = ["mean", "stderr"]

    # Case 2: scalar values
    elif df.shape[1] == 1:
        df.columns = ["value"]

    # Optional: split tuple index (m_2, m_3) into columns
    if isinstance(df.index[0], tuple):
        df.index = pd.MultiIndex.from_tuples(df.index, names=["m_2", "m_3"])
        df = df.reset_index()

    file_path = Path(folder) / f"{filename}.csv"
    df.to_csv(file_path, index=False)

    print(f"Saved to {file_path}")

if __name__ == "__main__":
    G = build_base_graph()

    solve_times = branch_solve_time_matrix(G, max_2=8, max_3=8, runs=4)
    save_dict_to_csv(solve_times, FOLDER, "branch_solve_times")
    plot_solve_time_matrix(solve_times)

    scenarios = create_scenarios(NUMBER_OF_STAGES, BRANCHES_PER_STAGE, G)

    # deviation_values = np.linspace(0, 1, 9) 
    # deviation_solve_times = allowed_deviation_solve_times(G, scenarios, deviation_values)
    # plot_deviation_solve_times(deviation_solve_times)

    cutting_plane_times = cutting_planes_solve_times(G, scenarios, 20, 20)
    save_dict_to_csv(cutting_plane_times, FOLDER, "cut_plane_solve_times")
    plot_cutting_plane_solve_times(cutting_plane_times)

    density_bounds = [1, 2, 3, 4]
    solve_times = density_solve_times(G, scenarios, density_bounds, runs = 4)
    save_dict_to_csv(solve_times, FOLDER, "density_solve_times")
    plot_density_solve_times(solve_times, runs=4)

    solve_times = splits_per_arc_experiment(G, splits_values=[6,11,16,21,26], runs= 4)

    plot_splits_solve_times(solve_times, runs= 4)
    save_dict_to_csv(solve_times, FOLDER, "splits_solve_times")
    plot_splits_objective(solve_times, runs= 4)



