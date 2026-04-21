import os
import sys
import time
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml


# Add parent directory
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import study_case_stochastic_model as scsm
import study_case_problem_file as scsp

# Load config
config_path = ROOT / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


# -----------------------------
# Default settings
# -----------------------------
NETWORKS = {
    "smaller_network": config['paths']['smaller_network_geojson'],
    "bigger_network": config['paths']['bigger_network_geojson'],
    "study_case_network": config['paths']['geojson_output'],
}

DATA_FOLDERS = {
    "bigger_network": config['paths']['compare_models_scenario_bigger'],
    "smaller_network": config['paths']['compare_models_scenario_smaller'],
    "study_case_network": config['paths']['compare_models_scenario_study_case'],
}

FIGURES_FOLDERS = {
    "bigger_network": config['paths']['compare_models_figures_bigger'],
    "smaller_network": config['paths']['compare_models_figures_smaller'],
    "study_case_network": config['paths']['compare_models_figures_study_case'],
}


# -----------------------------
# CSV logging
# -----------------------------
def init_csv(csv_path):
    """Create CSV with header if it doesn't exist."""
    if not os.path.exists(csv_path):
        # Ensure directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "network",
                "stage2",
                "stage3",
                "run",
                "seed",
                "solve_time"
            ])


def log_run(csv_path, row):
    """Append a single run to CSV immediately."""
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# -----------------------------
# Model runner
# -----------------------------
def run_model(geojson_file, data_folder, figure_folder, branches_per_stage, seed, n_stages):
    G = scsp.build_base_graph(geojson_file)

    scenarios = scsp.create_scenarios(
        n_stages,
        branches_per_stage,
        G,
        seed=seed,
        folder=data_folder
    )

    model = scsm.create_model(G, scenarios)

    start_time = time.time()
    results = scsm.solve_model(model, verbose=False, precision=0.001, threads =32)
    solve_time = time.time() - start_time

    scsm.save_model_values(model, figure_folder + f"model_run{seed}.pkl")

    return solve_time, results


# -----------------------------
# Main experiment logic
# -----------------------------
def run_experiment(args):
    init_csv(args.csv)

    results_summary = []

    for net_name in args.networks:
        geojson_file = NETWORKS[net_name]
        data_folder = DATA_FOLDERS[net_name]
        figure_folder = FIGURES_FOLDERS[net_name]

        print(f"\n=== Network: {net_name} ===")

        for stage2 in args.stage2:
            for stage3 in args.stage3:
                branches_per_stage = {1: 1, 2: stage2, 3: stage3}

                print(f"\nConfig: stage2={stage2}, stage3={stage3}")

                run_times = []

                for run in range(args.runs):
                    seed = args.base_seed + run

                    print(f"  Run {run+1}/{args.runs} (seed={seed})...")

                    solve_time, _ = run_model(
                        geojson_file,
                        data_folder,
                        figure_folder,
                        branches_per_stage,
                        seed,
                        args.stages
                    )

                    run_times.append(solve_time)

                    # 🔥 LOG IMMEDIATELY
                    log_run(args.csv, [
                        net_name,
                        stage2,
                        stage3,
                        run,
                        seed,
                        solve_time
                    ])

                    print(f"    -> {solve_time:.2f}s")

                mean_time = np.mean(run_times)
                std_time = np.std(run_times)

                results_summary.append({
                    "network": net_name,
                    "stage2": stage2,
                    "stage3": stage3,
                    "mean": mean_time,
                    "std": std_time
                })

                print(f"  ✅ Mean: {mean_time:.2f}s | Std: {std_time:.2f}s")

    return results_summary


# -----------------------------
# Plotting
# -----------------------------
def plot_results(results_summary, file_name):
    plt.figure(figsize=(10, 6))

    networks = sorted(set(r["network"] for r in results_summary))

    for net_name in networks:
        filtered = [r for r in results_summary if r["network"] == net_name]

        labels = [f"{r['stage2']}-{r['stage3']}" for r in filtered]
        means = [r["mean"] for r in filtered]
        stds = [r["std"] for r in filtered]

        plt.errorbar(labels, means, yerr=stds, marker='o', capsize=5, label=net_name)

    plt.xlabel("Stage2-Stage3 branching")
    plt.ylabel("Solve Time (s)")
    plt.title("Solve Time Comparison (Mean ± Std)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()


# -----------------------------
# Argument parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run stochastic model experiments")

    parser.add_argument("--runs", type=int, default=10, help="Number of repetitions per config")
    parser.add_argument("--stages", type=int, default=3, help="Number of stages")
    parser.add_argument("--stage2", nargs="+", type=int, default=[1], help="Stage 2 branching options")
    parser.add_argument("--stage3", nargs="+", type=int, default=[1], help="Stage 3 branching options")
    parser.add_argument("--networks", nargs="+", default=list(NETWORKS.keys()), help="Networks to test")
    parser.add_argument("--csv", type=str, default=FIGURES_FOLDERS["study_case_network"] +"experiment_results.csv", help="CSV log file")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed")

    return parser.parse_args()


# -----------------------------
# Entry point
# -----------------------------
def main():
    args = parse_args()

    results_summary = run_experiment(args)

    # Optional plotting
    if len(results_summary) > 0:
        plot_results(results_summary, file_name=FIGURES_FOLDERS["study_case_network"]+ "solve_times.png")


if __name__ == "__main__":
    main()