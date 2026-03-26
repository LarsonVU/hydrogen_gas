import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo
from pathlib import Path
import pandas as pd
import time
import os
import sys
import argparse

# =========================
# Path setup
# =========================
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from study_case_stochastic_model import solve_model, create_model, generate_cutting_plane_pairs, save_model_values
from study_case_problem_file import build_base_graph, create_scenarios

# =========================
# Argument parser
# =========================
parser = argparse.ArgumentParser(description="Density bounds impact experiment (CRN)")

parser.add_argument("--folder", type=str,
    default="study_case_model/figures/density_impact/",
    help="Output folder")

parser.add_argument("--branches_stage2", type=int, default=1)
parser.add_argument("--branches_stage3", type=int, default=1)

parser.add_argument("--precision", type=float, default=0.0001)

parser.add_argument("--runs", type=int, default=10)

args = parser.parse_args()

# =========================
# Settings
# =========================
FOLDER = args.folder
NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {1: 1, 2: args.branches_stage2, 3: args.branches_stage3}
PRECISION = args.precision
RUNS = args.runs

DENSITIES = [1, 5, 10]
SUBSIDIES = [0, 35, 40, 80]

# =========================
# Helpers
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


def compute_h2(model):
    values = [pyo.value(model.h2_production[m_3]) for m_3 in model.M[3]]
    return np.mean(values)


def compute_pressure_cost(model):
    values = [pyo.value(model.pressure_scenario[m_3]) for m_3 in model.M[3]]
    return np.mean(values)


# =========================
# Experiment
# =========================
def density_experiment_crn(G, densities, subsidies, runs):

    # Store raw results
    raw = {
        s: {d: {"h2": [], "pressure": []} for d in densities}
        for s in subsidies
    }

    for r in range(runs):
        print(f"\n===== RUN {r} =====")

        for subsidy in subsidies:
            print(f"--- Subsidy {subsidy} ---")

            sub_mwh = subsidy_per_mwh_to_mscm(subsidy)
            G_sub = apply_subsidy(G, sub_mwh)

            # CRN: same scenarios for all densities in THIS run
            scenarios = create_scenarios(
                NUMBER_OF_STAGES,
                BRANCHES_PER_STAGE,
                G_sub
            )

            for d in densities:
                print(f"Density {d}")

                model = create_model(
                    G_sub,
                    scenarios,
                    number_of_density_bounds=d,
                    cutting_plane_pairs=generate_cutting_plane_pairs(method="skewed"),
                    allowed_deviation=1                    
                )

                solve_model(model, verbose=False, precision=PRECISION, threads=16)
                save_model_values(model, FOLDER + f"model_pickles/model_run{r}_sub{subsidy}_den{d}" )

                h2 = compute_h2(model)
                pressure = compute_pressure_cost(model)

                raw[subsidy][d]["h2"].append(h2)
                raw[subsidy][d]["pressure"].append(pressure)
                print(f"Run {r} with subsidy {subsidy} and density {d}: h2 {h2}, pressure cost {pressure} ")

    # =========================
    # Aggregate
    # =========================
    results = {}

    for s in subsidies:
        results[s] = {}

        for d in densities:
            h2_vals = raw[s][d]["h2"]
            p_vals = raw[s][d]["pressure"]

            results[s][d] = {
                "h2_mean": np.mean(h2_vals),
                "h2_std": np.std(h2_vals),
                "pressure_mean": np.mean(p_vals),
                "pressure_std": np.std(p_vals),
            }

            print(
                f"Subsidy {s}, Density {d} | "
                f"H2: {results[s][d]['h2_mean']:.2f} ± {results[s][d]['h2_std']/np.sqrt(runs):.2f} | "
                f"P: {results[s][d]['pressure_mean']:.2f} ± {results[s][d]['pressure_std']/np.sqrt(runs):.2f}"
            )

    return results


# =========================
# Plotting
# =========================
def plot_results(results):
    Path(FOLDER).mkdir(parents=True, exist_ok=True)

    # ---- H2 ----
    plt.figure(figsize=(10, 6))

    for s in SUBSIDIES:
        means = [results[s][d]["h2_mean"] for d in DENSITIES]
        errs = [results[s][d]["h2_std"] / np.sqrt(RUNS) for d in DENSITIES]

        plt.errorbar(DENSITIES, means, yerr=errs, marker="o", capsize=5, label=f"Subsidy {s}")

    plt.xlabel("Density Bounds")
    plt.ylabel("Hydrogen Production")
    plt.title("H2 Production vs Density Bounds (CRN)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(FOLDER + "h2_vs_density.png")
    plt.close()

    # ---- Pressure ----
    plt.figure(figsize=(10, 6))

    for s in SUBSIDIES:
        means = [results[s][d]["pressure_mean"] for d in DENSITIES]
        errs = [results[s][d]["pressure_std"] / np.sqrt(RUNS) for d in DENSITIES]

        plt.errorbar(DENSITIES, means, yerr=errs, marker="o", capsize=5, label=f"Subsidy {s}")

    plt.xlabel("Density Bounds")
    plt.ylabel("Pressure Cost")
    plt.title("Pressure Cost vs Density Bounds (CRN)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(FOLDER + "pressure_vs_density.png")
    plt.close()


# =========================
# Save CSV
# =========================
def save_results(results):
    Path(FOLDER).mkdir(parents=True, exist_ok=True)
    rows = []

    for s in results:
        for d in results[s]:
            rows.append({
                "subsidy": s,
                "density": d,
                **results[s][d]
            })

    df = pd.DataFrame(rows)
    df.to_csv(Path(FOLDER) / "density_impact.csv", index=False)

    print("Saved CSV")


# =========================
# Run
# =========================
if __name__ == "__main__":
    G = build_base_graph()

    results = density_experiment_crn(G, DENSITIES, SUBSIDIES, RUNS)

    save_results(results)
    plot_results(results)