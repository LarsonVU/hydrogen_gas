import numpy as np
import pyomo.environ as pyo
import pandas as pd
import sys
import os
import argparse
from pathlib import Path
import itertools

# Add parent directory
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import study_case_stochastic_model as scsm
import study_case_problem_file as scpf
from Experiments.python_files.experiment_utils import subsidy_per_mwh_to_mscm, apply_subsidy


# =========================
# Argument parsing
# =========================
parser = argparse.ArgumentParser(description="Fixed scenario tree experiment")

parser.add_argument("--run", type=int, default=0)
parser.add_argument("--subsidy", type=float, default=70)
parser.add_argument("--deviation", type=float, default=0)

parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--time_limit", type=float, default=None)
parser.add_argument("--precision", type=float, default=0.01)
parser.add_argument("--upper_bounds", type=int, default=1)

parser.add_argument("--node_file_folder", type=str, default="node_files_fixed/")
parser.add_argument("--pickle_folder", type=str, default="study_case_model/figures/fixed_scenario_tree/model_values/")
parser.add_argument("--data_folder", type=str, default="study_case_model/scenario_variables/fixed_scenario_tree/")

args = parser.parse_args()


# =========================
# Constants
# =========================
NUMBER_OF_STAGES = 3

BRANCHES_PER_STAGE = {
    1: 1,
    2: 4,    # demand scenarios
    3: 16    # price scenarios
}


# =========================
# Group definitions
# =========================
PRICE_GROUPS = {
    "DUNKERQUE": "P1",
    "ZEEBRUGGE": "P2",
    "EMDEN": "P3",
    "DORNUM": "P3",
    "EASINGTON": "P4",
    "ST.FERGUS": "P4",
}

DEMAND_GROUPS = {
    "EASINGTON": "UK",
    "ST.FERGUS": "UK",
    "EMDEN": "DE",
    "DORNUM": "DE",
}


# =========================
# Helpers
# =========================
def generate_binary_patterns(n):
    return list(itertools.product(["low", "high"], repeat=n))


# =========================
# Fixed demand scenarios (Stage 2)
# =========================
def add_fixed_demand_scenarios(scenarios, deviation):
    patterns = generate_binary_patterns(2)  # UK, DE

    for i, scenario in enumerate(scenarios[2]):
        pattern = patterns[i % len(patterns)]

        demand_state = {
            "UK": pattern[0],
            "DE": pattern[1]
        }

        for node in scenario.G.nodes:
            node_data = scenario.G.nodes[node]

            if "average_demand_mwh_x1000" not in node_data:
                continue

            avg_demand = node_data.get("average_demand_mwh_x1000", 0) if node_data.get("average_demand_mwh_x1000", 0) is not None else 0
            var_demand = node_data.get("demand_variance", 0) if node_data.get("demand_variance", 0) is not None else 0
            avg = float(avg_demand) * 1000
            var = float(var_demand)

            group = DEMAND_GROUPS.get(node, None)

            if group is None:
                multiplier = 1.0
            else:
                if demand_state[group] == "high":
                    multiplier = 1 + deviation * var *2 
                else:
                    multiplier = 1 - deviation * var *2

            demand = max(avg * multiplier, 0)

            if "supplier_ratios" in node_data:
                ratios = node_data["supplier_ratios"] if node_data["supplier_ratios"] is not None else {}
                scenario.G.nodes[node]["demand"] = {
                    s: demand * r for s, r in ratios.items()
                }

    # Stage 3 inherits demand
    for scenario in scenarios[3]:
        pred = scenario.predecessor
        for node in scenario.G.nodes:
            if node in pred.G.nodes and "demand" in pred.G.nodes[node]:
                scenario.G.nodes[node]["demand"] = pred.G.nodes[node]["demand"]


# =========================
# Fixed price scenarios (Stage 3)
# =========================
def add_fixed_price_scenarios(scenarios, deviation):
    patterns = generate_binary_patterns(4)  # 4 price groups

    for i, scenario in enumerate(scenarios[3]):
        pattern = patterns[i % len(patterns)]

        price_state = {
            "P1": pattern[0],
            "P2": pattern[1],
            "P3": pattern[2],
            "P4": pattern[3],
        }

        for node in scenario.G.nodes:
            node_data = scenario.G.nodes[node]

            if "average_market_price" not in node_data:
                continue

            
            avg = float(node_data["average_market_price"] if node_data["average_market_price"] is not None else 0)
            std = float(node_data.get("long_term_price_std", 0) if node_data.get("long_term_price_std", 0) is not None else 0)

            group = PRICE_GROUPS.get(node, None)

            if group is None:
                continue

            if price_state[group] == "high":
                multiplier = 1 + deviation * std *2
            else:
                multiplier = 1 - deviation * std *2

            price = max(avg * multiplier, 0)

            scenario.G.nodes[node]["price"] = price


# =========================
# Scenario creation
# =========================
def create_fixed_scenarios(n_stages, b_stages, G, deviation):
    scenarios = {k: [] for k in range(1, n_stages + 1)}
    stage_probs = scpf.prob_per_stage(n_stages, b_stages)

    branches = 1
    for k in range(1, n_stages + 1):
        branches *= b_stages[k]
        for m in range(1, branches + 1):
            scenario = scpf.Scenario(
                k, m, stage_probs[(k, m)], G.copy(),
                predecessor=scenarios[k-1][int((m-1) // b_stages[k])] if k > 1 else None
            )
            scenarios[k].append(scenario)

    add_fixed_demand_scenarios(scenarios, deviation)
    add_fixed_price_scenarios(scenarios, deviation)

    scpf.add_generation_costs(scenarios, branches_per_stage=b_stages)
    scpf.add_booking_costs(scenarios, branches_per_stage=b_stages)

    return scenarios


# =========================
# MAIN
# =========================
def main():

    print("=== FIXED SCENARIO TREE EXPERIMENT ===")

    # Load base graph
    G = scpf.build_base_graph()

    # Apply subsidy if needed
    if args.subsidy != 0:
        subsidy_value = subsidy_per_mwh_to_mscm(args.subsidy)
        G = apply_subsidy(G, subsidy_value)

    # Create scenarios
    scenarios = create_fixed_scenarios(
        NUMBER_OF_STAGES,
        BRANCHES_PER_STAGE,
        G,
        args.deviation
    )

    print(f"Created {len(scenarios[3])} final scenarios")

    # Build model
    model = scsm.create_model(
        G,
        scenarios,
        allowed_deviation=args.deviation,
        number_of_density_bounds=args.upper_bounds
    )

    # Solve
    node_file_folder = args.node_file_folder
    os.makedirs(node_file_folder, exist_ok=True)

    try:
        results = scsm.solve_model(
            model,
            threads=args.threads,
            verbose=True,
            precision=args.precision,
            time_limit=args.time_limit,
            node_file_folder=node_file_folder
        )
    except Exception as e:
        print("Solve failed:", e)
        return

    print("Solve complete")

    # save results
    try:
        # Create pickle folder
        pickle_folder = os.path.join(
            args.pickle_folder,
            f"dev{args.deviation}",
            f"sub{args.subsidy}",
            f"run{args.run}"
        )
        scsm.save_model_values(model, filename =  os.path.join(pickle_folder, "model_snapshot.pkl"))
    except Exception as e:
        print("Saving failed:", e)

    print("=== DONE ===")


# =========================
# Run
# =========================
if __name__ == "__main__":
    main()