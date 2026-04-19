"""
N-k Contingency Analysis for Hydrogen Gas Network
===================================================

Tests network resilience under simultaneous infrastructure failures.
Goes beyond single-failure analysis by considering combinations of
pipe and/or plant failures (N-1, N-2, N-3 contingencies).

For each failure combination, solves the stochastic optimization model
and records key metrics: objective value, hydrogen production, demand
satisfaction, and feasibility status.

Usage:
    python examine_nk_contingency.py --max_k 2 --subsidy 30 --run 0
    python examine_nk_contingency.py --max_k 3 --subsidy 0 --run 0 --skip_plants

Output:
    Pickle files with model snapshots per failure combination.
    CSV summary of resilience metrics across all tested contingencies.
"""

import numpy as np
import pyomo.environ as pyo
import networkx as nx
import pandas as pd
import sys
import os
import argparse
import itertools
import json
from pathlib import Path

# Add parent directory
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import study_case_stochastic_model as scsm
import study_case_problem_file as scpf


# =========================
# Argument parsing
# =========================
parser = argparse.ArgumentParser(description="N-k contingency analysis for hydrogen gas network")

parser.add_argument("--run", type=int, default=0, help="Random seed / run index")
parser.add_argument("--branches_stage2", type=int, default=4)
parser.add_argument("--branches_stage3", type=int, default=4)
parser.add_argument("--subsidy", type=float, default=0, help="H2 subsidy in EUR/MWh")
parser.add_argument("--max_k", type=int, default=2,
                    help="Maximum number of simultaneous failures to test (1=N-1, 2=N-2, etc.)")
parser.add_argument("--skip_plants", action="store_true",
                    help="Only test pipe failures, skip plant failures")
parser.add_argument("--skip_pipes", action="store_true",
                    help="Only test plant failures, skip pipe failures")

parser.add_argument("--upper_bounds", type=int, default=1)
parser.add_argument("--time_limit", type=float, default=600, help="Solver time limit in seconds")
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--precision", type=float, default=0.01,
                    help="MIP gap (slightly relaxed for speed since we solve many instances)")

parser.add_argument("--data_folder", type=str, default="scenario_variables/nk_contingency/")
parser.add_argument("--pickle_folder", type=str, default="study_case_model/figures/nk_contingency/")
parser.add_argument("--output_csv", type=str, default="study_case_model/figures/nk_contingency/contingency_results.csv")

args = parser.parse_args()


# =========================
# Constants
# =========================
NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {1: 1, 2: args.branches_stage2, 3: args.branches_stage3}


# =========================
# Helper functions (shared pattern from existing experiments)
# =========================
def subsidy_per_mwh_to_mscm(mwh_subsidy, gcv_mwh_per_kscm=2.78):
    """Convert EUR/MWh subsidy to EUR/Mscm using hydrogen GCV."""
    return mwh_subsidy * gcv_mwh_per_kscm * 1000


def apply_subsidy(G, subsidy_value, variable_name="generation_cost"):
    """Reduce generation cost for hydrogen-producing nodes."""
    G_changed = G.copy()
    for node in G.nodes:
        if not pd.isna(G.nodes[node][variable_name]):
            if G.nodes[node]["component_ratio"]["H2"] > 0:
                G_changed.nodes[node][variable_name] = (
                    float(G.nodes[node][variable_name]) - subsidy_value
                )
    return G_changed


# ASCII-safe node name mapping (consistent with examine_network_failure.py)
NODE_NAME_MAP = {
    "GJØA": "GJOA",
    "VISUND": "VISUND",
    "NORNE ERB": "NORNE_ERB",
    "KÅRSTØ": "KARSTO",
    "DRAUPNER S": "DRAUPNER_S",
    "DORNUM": "DORNUM",
    "DUNKERQUE": "DUNKERQUE",
    "H-7 BP": "H-7_BP",
    "EMDEN": "EMDEN",
}


def sanitize_node_names(G):
    """Rename nodes to ASCII-safe names for compatibility."""
    G_safe = G.copy()
    node_mapping = {node: NODE_NAME_MAP.get(node, node) for node in G_safe.nodes()}
    G_safe = nx.relabel_nodes(G_safe, node_mapping)
    for u, v, data in G_safe.edges(data=True):
        for key in ("from_node", "to_node", "from", "to"):
            if key in data:
                data[key] = node_mapping.get(data[key], data[key])
    return G_safe


def get_failure_candidates(G):
    """
    Returns lists of (pipe_edges, plant_nodes) that can be failed.
    Pipes: all edges in the network.
    Plants: all generation/hydrogen nodes (those with supply_capacity).
    """
    pipes = list(G.edges())
    plants = [n for n, d in G.nodes(data=True)
              if pd.notna(d.get("supply_capacity", None))]
    return pipes, plants


def apply_failures(G, failed_pipes=None, failed_plants=None):
    """
    Apply a set of failures to the network.
    failed_pipes: list of (u, v) edge tuples to disable
    failed_plants: list of node names to disable
    """
    G_failed = G.copy()

    if failed_pipes:
        for u, v in failed_pipes:
            if (u, v) in G_failed.edges():
                G_failed.edges[u, v]["max_flow"] = 0

    if failed_plants:
        for node in failed_plants:
            if node in G_failed.nodes():
                G_failed.nodes[node]["supply_capacity"] = 0

    return G_failed


def generate_contingencies(pipes, plants, max_k, skip_pipes=False, skip_plants=False):
    """
    Generate all failure combinations up to max_k simultaneous failures.
    Returns list of dicts: [{"pipes": [...], "plants": [...], "k": int, "label": str}, ...]
    """
    contingencies = []

    # Baseline (no failure)
    contingencies.append({
        "pipes": [],
        "plants": [],
        "k": 0,
        "label": "baseline"
    })

    # Build pool of individual failure events
    failure_events = []
    if not skip_pipes:
        for pipe in pipes:
            failure_events.append(("pipe", pipe))
    if not skip_plants:
        for plant in plants:
            failure_events.append(("plant", plant))

    # Generate combinations up to max_k
    for k in range(1, min(max_k + 1, len(failure_events) + 1)):
        for combo in itertools.combinations(failure_events, k):
            failed_pipes = [item[1] for item in combo if item[0] == "pipe"]
            failed_plants = [item[1] for item in combo if item[0] == "plant"]

            label_parts = []
            for ftype, fitem in combo:
                if ftype == "pipe":
                    label_parts.append(f"pipe_{fitem[0]}_to_{fitem[1]}")
                else:
                    label_parts.append(f"plant_{fitem}")
            label = "__".join(label_parts)

            contingencies.append({
                "pipes": failed_pipes,
                "plants": failed_plants,
                "k": k,
                "label": label
            })

    return contingencies


def extract_metrics(model, results):
    """Extract key resilience metrics from a solved model."""
    metrics = {}

    term_cond = str(results.solver.termination_condition)
    metrics["termination_condition"] = term_cond
    metrics["feasible"] = term_cond in ("optimal", "feasible")

    if not metrics["feasible"]:
        metrics["objective"] = None
        metrics["avg_h2_production"] = None
        metrics["avg_revenue"] = None
        metrics["avg_demand_met_fraction"] = None
        return metrics

    try:
        metrics["objective"] = pyo.value(model.objective)

        # Average H2 production across scenarios
        h2_prod = []
        for m_3 in model.M[3]:
            h2_prod.append(pyo.value(model.h2_production[m_3]))
        metrics["avg_h2_production"] = np.mean(h2_prod)
        metrics["std_h2_production"] = np.std(h2_prod)

        # Average revenue
        revenues = []
        for m_3 in model.M[3]:
            revenues.append(pyo.value(model.revenue_scenario[m_3]))
        metrics["avg_revenue"] = np.mean(revenues)

        # Demand satisfaction: ratio of delivered energy to demanded energy
        demand_met_fracs = []
        for m_3 in model.M[3]:
            total_delivered = sum(
                pyo.value(model.gcv_c[c]) * pyo.value(model.f[a, c, m_3])
                for n in model.N_m
                for a in model.A_n_plus[n]
                for c in model.C
            )
            total_demanded = sum(
                pyo.value(model.D[h, n, m_3])
                for h in model.H
                for n in model.N_m
            )
            if total_demanded > 0:
                demand_met_fracs.append(total_delivered / total_demanded)
            else:
                demand_met_fracs.append(1.0)

        metrics["avg_demand_met_fraction"] = np.mean(demand_met_fracs)
        metrics["min_demand_met_fraction"] = np.min(demand_met_fracs)

    except Exception as e:
        print(f"Warning: could not extract metrics: {e}")
        metrics["objective"] = None

    return metrics


# =========================
# Main execution
# =========================
if __name__ == "__main__":
    # Build base graph
    G = scpf.build_base_graph()

    # Apply subsidy
    subsidy_mscm = subsidy_per_mwh_to_mscm(args.subsidy)
    G = apply_subsidy(G, subsidy_mscm)

    # Sanitize node names
    G = sanitize_node_names(G)

    # Get failure candidates
    pipes, plants = get_failure_candidates(G)
    print(f"Network has {len(pipes)} pipes and {len(plants)} plants available for failure testing.")

    # Generate contingencies
    contingencies = generate_contingencies(
        pipes, plants, args.max_k,
        skip_pipes=args.skip_pipes,
        skip_plants=args.skip_plants
    )
    print(f"Testing {len(contingencies)} contingency scenarios (baseline + up to N-{args.max_k}).")

    # Results collector
    all_results = []

    for i, contingency in enumerate(contingencies):
        label = contingency["label"]
        k = contingency["k"]
        print(f"\n[{i+1}/{len(contingencies)}] k={k}: {label}", flush=True)

        # Apply failures
        G_failed = apply_failures(G, contingency["pipes"], contingency["plants"])

        # Create scenario folder
        data_folder = os.path.join(
            args.data_folder,
            f"sub{args.subsidy}",
            f"run{args.run}",
            label
        )
        pickle_folder = os.path.join(
            args.pickle_folder,
            f"sub{args.subsidy}",
            f"run{args.run}",
            label
        )
        os.makedirs(data_folder, exist_ok=True)
        os.makedirs(pickle_folder, exist_ok=True)

        # Create scenarios (same seed for comparable results)
        scenarios = scpf.create_scenarios(
            NUMBER_OF_STAGES, BRANCHES_PER_STAGE, G_failed,
            seed=args.run, folder=data_folder
        )

        # Build and solve model
        model = scsm.create_model(
            G_failed, scenarios,
            number_of_density_bounds=args.upper_bounds
        )

        node_file_folder = os.environ.get("TMPDIR", "/tmp")
        node_file_folder = os.path.join(node_file_folder, f"gurobi_nk_{label}_run{args.run}")

        try:
            results = scsm.solve_model(
                model,
                threads=args.threads,
                verbose=False,
                precision=args.precision,
                time_limit=args.time_limit,
                node_file_folder=node_file_folder
            )

            # Extract metrics
            metrics = extract_metrics(model, results)

            # Save model snapshot
            if metrics["feasible"]:
                scsm.save_model_values(model, os.path.join(pickle_folder, "model_snapshot.pkl"))

        except Exception as e:
            print(f"  Solver failed: {e}")
            metrics = {"termination_condition": "error", "feasible": False,
                       "objective": None, "avg_h2_production": None}

        # Record results
        row = {
            "contingency_label": label,
            "k": k,
            "n_failed_pipes": len(contingency["pipes"]),
            "n_failed_plants": len(contingency["plants"]),
            "failed_pipes": str(contingency["pipes"]),
            "failed_plants": str(contingency["plants"]),
            "subsidy": args.subsidy,
            "run": args.run,
            **metrics
        }
        all_results.append(row)
        print(f"  Result: feasible={metrics['feasible']}, obj={metrics.get('objective', 'N/A')}")

    # Save summary CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")

    # Print summary table
    print("\n" + "=" * 80)
    print("CONTINGENCY ANALYSIS SUMMARY")
    print("=" * 80)
    for k_level in range(args.max_k + 1):
        subset = df[df["k"] == k_level]
        n_total = len(subset)
        n_feasible = subset["feasible"].sum()
        if n_feasible > 0:
            avg_obj = subset.loc[subset["feasible"], "objective"].mean()
            baseline_obj = df.loc[df["k"] == 0, "objective"].values[0] if df.loc[df["k"] == 0, "feasible"].values[0] else None
            loss_pct = ((baseline_obj - avg_obj) / abs(baseline_obj) * 100) if baseline_obj and avg_obj else "N/A"
        else:
            avg_obj = "N/A"
            loss_pct = "N/A"

        print(f"  N-{k_level}: {n_feasible}/{n_total} feasible, "
              f"avg objective = {avg_obj}, "
              f"avg loss vs baseline = {loss_pct}{'%' if isinstance(loss_pct, float) else ''}")

    print("\nFinished N-k contingency analysis!", flush=True)
