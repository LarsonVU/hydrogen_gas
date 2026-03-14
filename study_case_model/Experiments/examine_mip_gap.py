from study_case_stochastic_model import solve_model, create_model, generate_cutting_plane_pairs
from study_case_problem_file import build_base_graph,  create_scenarios
import matplotlib.pyplot as plt
from pyomo.opt import TerminationCondition
import numpy as np
import time
import os

FOLDER = "study_case_model/figures/mip_gap/"
NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {1 : 1, 2 : 2, 3: 4}

def solve_gap_after_time(model, allowed_time=100):
    results = solve_model(model, verbose=False, time_limit=allowed_time)

    gap = None
    if hasattr(results.solver, "mip_gap"):
        gap = results.solver.mip_gap
    elif hasattr(results.problem, "upper_bound") and hasattr(results.problem, "lower_bound"):
        ub = results.problem.upper_bound
        lb = results.problem.lower_bound
        if ub is not None and lb is not None and ub != 0:
            gap = abs(ub - lb) / abs(ub)

    return gap

def branch_gap_matrix(network, max_2, max_3, max_time =100):
    G = network.copy()

    gaps = {}

    for m_2 in range(1, max_2+1):
        for m_3 in range(1, max_3+1):
            scenarios = create_scenarios(NUMBER_OF_STAGES, {1: 1, 2: m_2, 3: m_3}, G)
            model = create_model(G, scenarios)
            gaps[(m_2, m_3)] = solve_gap_after_time(model, allowed_time=max_time)
            print(f"Branches Stage 2: {m_2}, Branches Stage 3: {m_3}, Remaining gap: {gaps[(m_2, m_3)]} percent")
    return gaps
    
def plot_gap_matrix(gaps, folder =FOLDER):
    os.makedirs(folder, exist_ok=True)
    m_2_values = sorted(set(m_2 for m_2, m_3 in gaps.keys()))
    m_3_values = sorted(set(m_3 for m_2, m_3 in gaps.keys()))

    Z = np.array([[gaps.get((m_2, m_3), 0) for m_3 in m_3_values] for m_2 in m_2_values])
    plt.figure(figsize=(10, 6))
    plt.imshow(Z, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='MIP Gap')
    for i, m_2 in enumerate(m_2_values):
        for j, m_3 in enumerate(m_3_values):
            plt.text(j, i, f'{Z[i, j]:.2f}', ha='center', va='center', color='white', fontsize=10)
    plt.xticks(ticks=range(len(m_3_values)), labels=m_3_values)
    plt.yticks(ticks=range(len(m_2_values)), labels=m_2_values)
    plt.xlabel('Branches in Stage 3')
    plt.ylabel('Branches in Stage 2')
    plt.title(f'MIP Gap Matrix')
    plt.tight_layout()
    plt.savefig(folder +'mip_gap_matrix.png')
    plt.show()

def allowed_deviation_gaps(network, scenarios, deviation_values, max_time=100):
    G = network.copy()
    gaps = {}

    for deviation in deviation_values:
        model = create_model(G, scenarios, allowed_deviation=deviation)
        gap = solve_gap_after_time(model, allowed_time=max_time)

        gaps[deviation] = gap
        print(f"Allowed Deviation: {deviation}, MIP Gap: {gap}")

    return gaps

def plot_deviation_gaps(gaps, max_time):
    os.makedirs(FOLDER, exist_ok=True)
    deviations = list(gaps.keys())
    gap_values = list(gaps.values())

    plt.figure(figsize=(10, 6))
    plt.plot(deviations, gap_values, marker='o')
    plt.xlabel('Allowed Deviation')
    plt.ylabel('MIP Gap')
    plt.title(f'MIP Gap vs Allowed Deviation (Time Limit = {max_time}s)')
    plt.grid()
    plt.tight_layout()
    plt.savefig(FOLDER + 'deviation_mip_gaps.png')
    plt.show()

def cutting_planes_gaps(network, scenarios,
                        max_cutting_planes_in,
                        max_cutting_planes_out,
                        max_time=100):

    G = network.copy()
    gaps = {}

    for in_planes in range(1, max_cutting_planes_in + 1):
        for out_planes in range(1, max_cutting_planes_out + 1):

            cutting_plane_pairs = generate_cutting_plane_pairs(
                n_p_in=in_planes,
                n_p_out=out_planes
            )

            model = create_model(G, scenarios,
                                 cutting_plane_pairs=cutting_plane_pairs)

            gap = solve_gap_after_time(model, allowed_time=max_time)

            gaps[(in_planes, out_planes)] = gap

            print(f"In: {in_planes}, Out: {out_planes}, MIP Gap: {gap}")

    return gaps

def plot_cutting_plane_gaps(gaps):
    os.makedirs(FOLDER, exist_ok=True)
    in_vals = sorted(set(i for i, _ in gaps.keys()))
    out_vals = sorted(set(o for _, o in gaps.keys()))

    Z = np.array([
        [gaps.get((i, o), 0) for o in out_vals]
        for i in in_vals
    ])

    plt.figure(figsize=(10, 6))
    plt.imshow(Z, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='MIP Gap')

    for i, in_val in enumerate(in_vals):
        for j, out_val in enumerate(out_vals):
            plt.text(j, i, f'{Z[i, j]:.3f}',
                     ha='center', va='center', color='white')

    plt.xticks(range(len(out_vals)), out_vals)
    plt.yticks(range(len(in_vals)), in_vals)
    plt.xlabel('Number of Cutting Planes Out')
    plt.ylabel('Number of Cutting Planes In')
    plt.title('MIP Gap by Cutting Plane Counts')
    plt.tight_layout()
    plt.savefig(FOLDER + 'cutting_plane_mip_gaps.png')
    plt.show()

if __name__ == "__main__":
    G = build_base_graph()
    # gaps= branch_gap_matrix(G, 5, 5, 40)
    # plot_gap_matrix(gaps, FOLDER)

    deviation_values = np.linspace(0, 1, 9) 
    scenarios = create_scenarios(NUMBER_OF_STAGES, BRANCHES_PER_STAGE, G)
    max_time =40

    # gaps_dev = allowed_deviation_gaps(G, scenarios, deviation_values, max_time=max_time)
    # plot_deviation_gaps(gaps_dev, max_time)

    max_cut_plane_in = 5
    max_cut_plane_out = 5

    gaps_plane = cutting_planes_gaps(G, scenarios, max_cut_plane_in, max_cut_plane_out, max_time= max_time)
    plot_cutting_plane_gaps(gaps_plane)