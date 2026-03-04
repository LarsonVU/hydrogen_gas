from study_case_stochastic_model import solve_model, create_model, generate_cutting_plane_pairs
from study_case_problem_file import build_base_graph,  create_scenarios
import matplotlib.pyplot as plt
from pyomo.opt import TerminationCondition
import numpy as np
import pyomo.environ as pyo
import time
import os

NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {1 : 1, 2 : 2, 3: 4}

def compute_weymouth_deviation(model):
    """
    Computes:
        - max absolute percentage deviation
        - average absolute percentage deviation
        - full deviation dictionary

    Deviation formula:
        |total_flow - weymouth_flow| / |total_flow|
    """

    deviations = []
    deviation_dict = {}

    for a in model.A:
        for m_3 in model.M[3]:

            total = float(pyo.value(model.total_flow[a, m_3]))
            weymouth = float(pyo.value(model.weymouth_flow[a, m_3]))

            if abs(total) > 1e-8:
                deviation = abs(total - weymouth) / abs(total)
            else:
                deviation = 0.0

            deviations.append(deviation)
            deviation_dict[(a, m_3)] = deviation

    deviations = np.array(deviations)

    max_dev = np.max(deviations)
    avg_dev = np.mean(deviations)

    return max_dev, avg_dev, deviation_dict


def solve_and_compute_deviation(model, time_limit=None):
    """
    Solve model and compute deviation afterwards.
    """

    print("Solving model...")
    solve_model(model, verbose=False, time_limit=time_limit)

    print("Computing Weymouth deviations...")
    max_dev, avg_dev, deviation_dict = compute_weymouth_deviation(model)

    print(f"Maximum deviation: {max_dev :.4f}%")
    print(f"Average deviation: {avg_dev :.4f}%")

    return max_dev, avg_dev, deviation_dict

def cutting_planes_deviation(network, scenarios,
                              cutting_planes_in,
                              cutting_planes_out,
                              max_time=100):

    G = network.copy()

    # Store max deviation per (in_planes, out_planes)
    deviations = {}

    for in_planes in cutting_planes_in:
        for out_planes in cutting_planes_out:
            print(f"Running: In={in_planes}, Out={out_planes}")

            cutting_plane_pairs = generate_cutting_plane_pairs(
                n_p_in=in_planes,
                n_p_out=out_planes
            )

            model = create_model(
                G,
                scenarios,
                cutting_plane_pairs=cutting_plane_pairs
            )

            # 👇 Solve and compute deviation instead of MIP gap
            max_dev, avg_dev, deviation_dict = solve_and_compute_deviation(
                model,
                time_limit=max_time
            )

            # Store results
            deviations[(in_planes, out_planes)] = {
                "max_deviation": max_dev,
                "avg_deviation": avg_dev,
                "all_deviations": deviation_dict,
            }

            print(f"Stored max deviation: {max_dev :.4f}%\n")

    return deviations

def plot_cutting_plane_deviation(deviations, folder):
    """
    deviations:
        dict with structure:
        {(in_planes, out_planes): {
            "max_deviation": float,
            "avg_deviation": float,
            ...
        }}
    """

    os.makedirs(folder, exist_ok=True)

    in_vals = sorted(set(i for i, _ in deviations.keys()))
    out_vals = sorted(set(o for _, o in deviations.keys()))

    # --- MAX deviation matrix ---
    Z_max = np.array([
        [
            deviations.get((i, o), {}).get("max_deviation", 0) 
            for o in out_vals
        ]
        for i in in_vals
    ])

    # --- AVG deviation matrix ---
    Z_avg = np.array([
        [
            deviations.get((i, o), {}).get("avg_deviation", 0) 
            for o in out_vals
        ]
        for i in in_vals
    ])

    # ============================
    # Plot 1: Maximum deviation
    # ============================
    plt.figure(figsize=(10, 6))
    im = plt.imshow(Z_max, origin='lower', aspect='auto', cmap='viridis')

    cbar = plt.colorbar(im)
    cbar.set_label('Max Deviation (%)')

    for i in range(len(in_vals)):
        for j in range(len(out_vals)):
            plt.text(j, i,
                     f'{Z_max[i, j]:.2f}',
                     ha='center',
                     va='center',
                     color='white')

    plt.xticks(range(len(out_vals)), out_vals)
    plt.yticks(range(len(in_vals)), in_vals)
    plt.xlabel('Number of Cutting Planes Out')
    plt.ylabel('Number of Cutting Planes In')
    plt.title('Maximum Weymouth Deviation (%)')

    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'cutting_plane_max_deviation.png'))
    plt.show()

    # ============================
    # Plot 2: Average deviation
    # ============================
    plt.figure(figsize=(10, 6))
    im = plt.imshow(Z_avg, origin='lower', aspect='auto', cmap='viridis')

    cbar = plt.colorbar(im)
    cbar.set_label('Average Deviation (%)')

    for i in range(len(in_vals)):
        for j in range(len(out_vals)):
            plt.text(j, i,
                     f'{Z_avg[i, j]:.2f}',
                     ha='center',
                     va='center',
                     color='white')

    plt.xticks(range(len(out_vals)), out_vals)
    plt.yticks(range(len(in_vals)), in_vals)
    plt.xlabel('Number of Cutting Planes Out')
    plt.ylabel('Number of Cutting Planes In')
    plt.title('Average Weymouth Deviation (%)')

    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'cutting_plane_avg_deviation.png'))
    plt.show()

if __name__ == "__main__":

    print("\nBuilding base graph...")
    G = build_base_graph()

    print("Creating scenarios...")
    scenarios = create_scenarios(
        NUMBER_OF_STAGES,
        BRANCHES_PER_STAGE,
        G
    )

    max_cut_plane_in = [5 + i for i in range(5)]
    max_cut_plane_out = [5 + i for i in range(5)]
    max_time = 100  # seconds per solve

    print("\nStarting cutting-plane deviation experiment...\n")

    deviations = cutting_planes_deviation(
        network=G,
        scenarios=scenarios,
        cutting_planes_in=max_cut_plane_in,
        cutting_planes_out=max_cut_plane_out,
        max_time=max_time
    )

    print("\nPlotting results...")
    plot_cutting_plane_deviation(
        deviations,
        folder="figures/"
    )

    # -------------------------------------------------
    # Print best configuration (minimum max deviation)
    # -------------------------------------------------
    best_config = min(
        deviations.items(),
        key=lambda x: x[1]["max_deviation"]
    )

    (best_in, best_out), best_values = best_config

    print("\nBest Cutting Plane Configuration:")
    print(f"In planes  : {best_in}")
    print(f"Out planes : {best_out}")
    print(f"Max deviation: {best_values['max_deviation']:.4f}")
    print(f"Avg deviation: {best_values['avg_deviation']:.4f}")