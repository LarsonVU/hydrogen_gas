from stochastic_model import solve_model, create_model, generate_cutting_plane_pairs
from problem_file import build_base_graph,  create_scenarios
import matplotlib.pyplot as plt
import numpy as np
import time

# Folder
FOLDER = "results/solve_times/"

# Base Parameters
NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {1 : 1, 2 : 2, 3: 4}
PRECISION = 0.001

def time_model(model):
    start_time = time.time()
    results = solve_model(model, verbose=False, precision=PRECISION)
    end_time = time.time()
    return end_time - start_time


def branch_solve_time_matrix(network, max_2, max_3):
    G = network.copy()

    solve_times = {}

    for m_2 in range(1, max_2+1):
        for m_3 in range(1, max_3+1):
            scenarios = create_scenarios(NUMBER_OF_STAGES, {1: 1, 2: m_2, 3: m_3}, G)
            model = create_model(G, scenarios)
            solve_times[(m_2, m_3)] = time_model(model)
            print(f"Branches Stage 2: {m_2}, Branches Stage 3: {m_3}, Solve Time: {solve_times[(m_2, m_3)]} seconds")
    return solve_times
    


def plot_solve_time_matrix(solve_times):
    m_2_values = sorted(set(m_2 for m_2, m_3 in solve_times.keys()))
    m_3_values = sorted(set(m_3 for m_2, m_3 in solve_times.keys()))

    Z = np.array([[solve_times.get((m_2, m_3), 0) for m_3 in m_3_values] for m_2 in m_2_values])
    plt.figure(figsize=(10, 6))
    plt.imshow(Z, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Solve Time (seconds)')
    for i, m_2 in enumerate(m_2_values):
        for j, m_3 in enumerate(m_3_values):
            plt.text(j, i, f'{Z[i, j]:.2f}', ha='center', va='center', color='white', fontsize=10)
    plt.xticks(ticks=range(len(m_3_values)), labels=m_3_values)
    plt.yticks(ticks=range(len(m_2_values)), labels=m_2_values)
    plt.xlabel('Branches in Stage 3')
    plt.ylabel('Branches in Stage 2')
    plt.title(f'Solve Time Matrix (Tolerance = {PRECISION *100}%)')
    plt.tight_layout()
    plt.savefig(FOLDER +'solve_time_matrix.png')
    plt.show()


def allowed_deviation_solve_times(network, scenarios, deviation_values):
    G = network.copy()
    solve_times = {}

    for deviation in deviation_values:
        model = create_model(G, scenarios, allowed_deviation=deviation)
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
    plt.show()

def cutting_planes_solve_times(network, scenarios, max_cutting_planes_in, max_cutting_planes_out):
    G = network.copy()
    cutting_plane_counts = {}

    for in_planes in range(0, max_cutting_planes_in + 1):
        for out_planes in range(0, max_cutting_planes_out + 1):
            cutting_plane_pairs = generate_cutting_plane_pairs(n_p_in=in_planes, n_p_out=out_planes)
            model = create_model(G, scenarios, cutting_plane_pairs=cutting_plane_pairs)
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
    plt.show()

if __name__ == "__main__":
    G = build_base_graph()

    # solve_times = branch_solve_time_matrix(G, max_2=5, max_3=5)
    # plot_solve_time_matrix(solve_times)

    deviation_values = np.linspace(0, 1, 9) 
    scenarios = create_scenarios(NUMBER_OF_STAGES, BRANCHES_PER_STAGE, G)

    deviation_solve_times = allowed_deviation_solve_times(G, scenarios, deviation_values)
    plot_deviation_solve_times(deviation_solve_times)


