"""
Value of Stochastic Solution (VSS) Experiment
===============================================

Computes the Value of the Stochastic Solution, a standard metric in
stochastic programming that quantifies the benefit of modeling uncertainty
explicitly versus using a deterministic (expected-value) approach.

The VSS is computed as:

    VSS = RP - EEV

where:
    RP  = "Recourse Problem" value = optimal objective of the full stochastic model
    EEV = "Expected result of using the Expected Value solution"
        = solve a deterministic model with mean parameters, then fix the
          first-stage decisions and evaluate against all scenarios

In practice, this means:
1. Solve the full stochastic model → get RP
2. Solve a deterministic model (1 scenario with mean demand/prices) → get EV solution
3. Fix the first-stage (booking) decisions from step 2, re-solve the stochastic
   model with those fixed decisions → get EEV
4. VSS = RP - EEV

A large positive VSS means the stochastic formulation adds significant value.
A near-zero VSS means the deterministic approach is nearly as good, and the
added complexity of stochastic programming may not be justified.

Usage:
    python examine_vss.py --run 0 --subsidy 30
    python examine_vss.py --run 0 --subsidy 0 --branches_stage2 8 --branches_stage3 8

Output:
    CSV with RP, EV, EEV, and VSS values.
"""

import numpy as np
import pyomo.environ as pyo
import networkx as nx
import pandas as pd
import sys
import os
import argparse
import copy
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
parser = argparse.ArgumentParser(description="Value of Stochastic Solution experiment")

parser.add_argument("--run", type=int, default=0)
parser.add_argument("--branches_stage2", type=int, default=8)
parser.add_argument("--branches_stage3", type=int, default=8)
parser.add_argument("--subsidy", type=float, default=70)
parser.add_argument("--deviation", type=float, default=0.0)

parser.add_argument("--upper_bounds", type=int, default=1)
parser.add_argument("--time_limit", type=float, default=None)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--precision", type=float, default=0.002)

parser.add_argument("--data_folder", type=str, default="scenario_variables/vss/")
parser.add_argument("--pickle_folder", type=str, default="study_case_model/figures/vss/")
parser.add_argument("--output_csv", type=str, default="study_case_model/figures/vss/vss_results.csv")

# Additional runs for statistical significance
parser.add_argument("--n_runs", type=int, default=1,
                    help="Number of independent runs (each with different seed)")

args = parser.parse_args()


# =========================
# Constants
# =========================
NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {1: 1, 2: args.branches_stage2, 3: args.branches_stage3}




def create_deterministic_scenarios(G, folder):
    """
    Create a 'deterministic' scenario set: single scenario per stage with
    mean demand and mean prices (no stochastic sampling).

    This represents the Expected Value (EV) problem.
    """
    os.makedirs(folder, exist_ok=True)

    # Use branches = 1 at all stages → single path through the tree
    det_branches = {1: 1, 2: 1, 3: 1}
    n_stages = NUMBER_OF_STAGES

    scenarios = {k: [] for k in range(1, n_stages + 1)}
    stage_probs = scpf.prob_per_stage(n_stages, det_branches)

    branches = 1
    for k in range(1, n_stages + 1):
        branches = det_branches[k] * branches
        for m in range(1, branches + 1):
            scenario = scpf.Scenario(
                k, m, stage_probs[(k, m)], G.copy(),
                predecessor=scenarios[k-1][int((m-1) // det_branches[k])] if k > 1 else None
            )
            scenarios[k].append(scenario)

    # Set demand to average (no sampling) for all market nodes
    for scenario in scenarios[2]:
        for node in scenario.G.nodes:
            node_data = scenario.G.nodes[node]
            if "average_demand_mwh_x1000" in node_data and not pd.isna(node_data["average_demand_mwh_x1000"]):
                avg_demand = float(node_data["average_demand_mwh_x1000"]) * 1000
                if "supplier_ratios" in node_data and not pd.isna(node_data["supplier_ratios"]):
                    supplier_ratios = node_data["supplier_ratios"][0] if isinstance(node_data["supplier_ratios"], list) else node_data["supplier_ratios"]
                    scenario.G.nodes[node]["demand"] = {
                        supplier: avg_demand * ratio
                        for supplier, ratio in supplier_ratios.items()
                    }

    # Stage 3 inherits from stage 2
    for scenario in scenarios[3]:
        predecessor = scenario.predecessor
        for node in scenario.G.nodes:
            if node in predecessor.G.nodes and "demand" in predecessor.G.nodes[node]:
                scenario.G.nodes[node]["demand"] = predecessor.G.nodes[node]["demand"]

    # Set prices to average (no sampling)
    for scenario in scenarios[2]:
        for node in scenario.G.nodes:
            node_data = scenario.G.nodes[node]
            if "average_market_price" in node_data and not pd.isna(node_data["average_market_price"]):
                scenario.G.nodes[node]["price"] = float(node_data["average_market_price"])

    for scenario in scenarios[3]:
        predecessor = scenario.predecessor
        for node in scenario.G.nodes:
            if node in predecessor.G.nodes and "price" in predecessor.G.nodes[node]:
                scenario.G.nodes[node]["price"] = predecessor.G.nodes[node]["price"]
            elif "average_market_price" in scenario.G.nodes[node] and not pd.isna(scenario.G.nodes[node].get("average_market_price")):
                scenario.G.nodes[node]["price"] = float(scenario.G.nodes[node]["average_market_price"])
            else:
                scenario.G.nodes[node]["price"] = 0

    # Booking costs (same logic as base model)
    scpf.add_generation_costs(scenarios, branches_per_stage=det_branches)
    scpf.add_booking_costs(scenarios, branches_per_stage=det_branches)

    return scenarios


def fix_first_stage_decisions(stoch_model, ev_model):
    """
    Fix the first-stage (stage 1 and stage 2) booking decisions in the
    stochastic model to the values from the EV solution.

    The key non-anticipativity insight: in the EV model there's only one
    scenario, so the booking decisions are deterministic. We fix these
    same booking levels across all scenarios in the stochastic model.

    Returns the modified stochastic model.
    """
    # Get EV booking decisions (stage 1 and stage 2)
    ev_entry_1 = {n: pyo.value(ev_model.x_entry[n, 1, 1]) for n in ev_model.N}
    ev_exit_1 = {n: pyo.value(ev_model.x_exit[n, 1, 1]) for n in ev_model.N}
    ev_entry_2 = {n: pyo.value(ev_model.x_entry[n, 2, 1]) for n in ev_model.N}
    ev_exit_2 = {n: pyo.value(ev_model.x_exit[n, 2, 1]) for n in ev_model.N}

    # Fix stage 1 decisions in stochastic model (same for all scenarios)
    for n in stoch_model.N:
        stoch_model.x_entry[n, 1, 1].fix(ev_entry_1.get(n, 0))
        stoch_model.x_exit[n, 1, 1].fix(ev_exit_1.get(n, 0))

    # Fix stage 2 decisions (apply same EV decision to ALL stage-2 scenarios)
    for n in stoch_model.N:
        for m_2 in stoch_model.M[2]:
            stoch_model.x_entry[n, 2, m_2].fix(ev_entry_2.get(n, 0))
            stoch_model.x_exit[n, 2, m_2].fix(ev_exit_2.get(n, 0))

    return stoch_model


def build_single_path_scenarios(leaf_scenario):
    """Build a single deterministic scenario tree for one stage-3 path."""
    if leaf_scenario.stage != 3:
        raise ValueError("leaf_scenario must be a stage-3 leaf")

    stage2 = leaf_scenario.predecessor
    if stage2 is None or stage2.predecessor is None:
        raise ValueError("leaf_scenario is missing its predecessor chain")

    stage1 = stage2.predecessor

    scenario1 = scpf.Scenario(1, 1, 1.0, stage1.G.copy())
    scenario2 = scpf.Scenario(2, 1, 1.0, stage2.G.copy(), predecessor=scenario1)
    scenario3 = scpf.Scenario(3, 1, 1.0, leaf_scenario.G.copy(), predecessor=scenario2)

    return {1: [scenario1], 2: [scenario2], 3: [scenario3]}


def compute_evpi(network, leaf_scenarios, allowed_deviation, number_of_density_bounds, threads, precision, time_limit, seed):
    """Compute the expected value of perfect information (EVPI)."""
    evpi = 0.0
    total_prob = 0.0

    for leaf in leaf_scenarios:
        deterministic_scenarios = build_single_path_scenarios(leaf)
        model_ws = scsm.create_model(
            network, deterministic_scenarios,
            allowed_deviation=allowed_deviation,
            number_of_density_bounds=number_of_density_bounds
        )

        node_file = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"gurobi_evpi_run{seed}_scenario{leaf.index}")
        try:
            scsm.solve_model(
                model_ws, threads=threads, verbose=False,
                precision=precision, time_limit=time_limit,
                node_file_folder=node_file
            )
            objective = pyo.value(model_ws.objective)
            if objective is None:
                raise ValueError("Objective is None after solve")

            evpi += leaf.probability * objective
            total_prob += leaf.probability
        except Exception as e:
            print(f"  WARNING: EVPI solve failed for leaf scenario {leaf.index}: {e}")
            return None

    if not np.isclose(total_prob, 1.0):
        print(f"  WARNING: total leaf probability = {total_prob:.6f} (expected 1.0)")

    return evpi


def extract_vss_metrics(model, label):
    """Extract objective and key metrics from a solved model."""
    metrics = {"model_type": label}
    try:
        metrics["objective"] = pyo.value(model.objective)

        obj_vals = [pyo.value(model.scenario_objective[m_3]) for m_3 in model.M[3]]
        metrics["avg_scenario_obj"] = np.mean(obj_vals)
        metrics["std_scenario_obj"] = np.std(obj_vals)
        metrics["worst_scenario_obj"] = np.min(obj_vals)

        h2_vals = [pyo.value(model.h2_production[m_3]) for m_3 in model.M[3]]
        metrics["avg_h2_production"] = np.mean(h2_vals)

        # Total booking
        metrics["total_stage1_entry"] = sum(pyo.value(model.x_entry[n, 1, 1]) for n in model.N)
        metrics["total_stage1_exit"] = sum(pyo.value(model.x_exit[n, 1, 1]) for n in model.N)

    except Exception as e:
        print(f"Warning: could not extract metrics for {label}: {e}")
        metrics["objective"] = None

    return metrics


# =========================
# Main execution
# =========================
if __name__ == "__main__":
    all_results = []

    for run_idx in range(args.n_runs):
        seed = args.run + run_idx
        print(f"\n{'='*70}")
        print(f"RUN {run_idx + 1}/{args.n_runs} (seed={seed})")
        print(f"{'='*70}")

        # Build base graph
        G = scpf.build_base_graph()
        subsidy_mscm = subsidy_per_mwh_to_mscm(args.subsidy)
        G = apply_subsidy(G, subsidy_mscm)

        # ---- Step 1: Solve full stochastic model (RP) ----
        print("\n[Step 1] Solving full stochastic model (Recourse Problem)...")
        data_folder_rp = os.path.join(args.data_folder, f"rp/sub{args.subsidy}/run{seed}")
        pickle_folder_rp = os.path.join(args.pickle_folder, f"rp/sub{args.subsidy}/run{seed}")
        os.makedirs(data_folder_rp, exist_ok=True)
        os.makedirs(pickle_folder_rp, exist_ok=True)

        scenarios_rp = scpf.create_scenarios(
            NUMBER_OF_STAGES, BRANCHES_PER_STAGE, G,
            seed=seed, folder=data_folder_rp
        )

        model_rp = scsm.create_model(
            G, scenarios_rp,
            allowed_deviation=args.deviation,
            number_of_density_bounds=args.upper_bounds
        )

        node_file = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"gurobi_vss_rp_run{seed}")
        results_rp = scsm.solve_model(
            model_rp, threads=args.threads, verbose=True,
            precision=args.precision, time_limit=args.time_limit,
            node_file_folder=node_file
        )

        rp_metrics = extract_vss_metrics(model_rp, "RP")
        RP = rp_metrics.get("objective")
        print(f"  RP objective: {RP}")

        if RP is None:
            print("  Stochastic model infeasible — skipping this run.")
            continue

        scsm.save_model_values(model_rp, os.path.join(pickle_folder_rp, "model_snapshot.pkl"))

        # ---- EVPI: Solve each leaf deterministically and average ----
        print("\n[EVPI] Solving each scenario deterministically for perfect information...")
        EV_PI = compute_evpi(
            G, scenarios_rp[3],
            allowed_deviation=args.deviation,
            number_of_density_bounds=args.upper_bounds,
            threads=args.threads,
            precision=args.precision,
            time_limit=args.time_limit,
            seed=seed
        )
        EVPI = None if EV_PI is None or RP is None else EV_PI - RP
        print(f"  EV_PI expected objective: {EV_PI}")
        print(f"  EVPI (EV_PI - RP): {EVPI}")

        # ---- Step 2: Solve deterministic (EV) model ----
        print("\n[Step 2] Solving deterministic Expected Value model...")
        data_folder_ev = os.path.join(args.data_folder, f"ev/sub{args.subsidy}/run{seed}")
        pickle_folder_ev = os.path.join(args.pickle_folder, f"ev/sub{args.subsidy}/run{seed}")
        os.makedirs(data_folder_ev, exist_ok=True)
        os.makedirs(pickle_folder_ev, exist_ok=True)

        scenarios_ev = create_deterministic_scenarios(G, folder=data_folder_ev)

        model_ev = scsm.create_model(
            G, scenarios_ev,
            allowed_deviation=0,  # No deviation needed for deterministic
            number_of_density_bounds=args.upper_bounds
        )

        node_file = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"gurobi_vss_ev_run{seed}")
        results_ev = scsm.solve_model(
            model_ev, threads=args.threads, verbose=False,
            precision=args.precision, time_limit=args.time_limit,
            node_file_folder=node_file
        )

        ev_metrics = extract_vss_metrics(model_ev, "EV")
        EV = ev_metrics.get("objective")
        print(f"  EV objective: {EV}")

        if EV is None:
            print("  EV model infeasible — skipping this run.")
            continue

        scsm.save_model_values(model_ev, os.path.join(pickle_folder_ev, "model_snapshot.pkl"))

        # ---- Step 3: Fix EV decisions, re-solve stochastic (EEV) ----
        print("\n[Step 3] Solving EEV (stochastic model with fixed EV decisions)...")
        data_folder_eev = os.path.join(args.data_folder, f"eev/sub{args.subsidy}/run{seed}")
        pickle_folder_eev = os.path.join(args.pickle_folder, f"eev/sub{args.subsidy}/run{seed}")
        os.makedirs(data_folder_eev, exist_ok=True)
        os.makedirs(pickle_folder_eev, exist_ok=True)

        # Re-create stochastic scenarios with SAME seed for comparability
        scenarios_eev = scpf.create_scenarios(
            NUMBER_OF_STAGES, BRANCHES_PER_STAGE, G,
            seed=seed, folder=data_folder_eev
        )

        model_eev = scsm.create_model(
            G, scenarios_eev,
            allowed_deviation=args.deviation,
            number_of_density_bounds=args.upper_bounds
        )

        # Fix first-stage decisions from EV solution
        model_eev = fix_first_stage_decisions(model_eev, model_ev)

        node_file = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"gurobi_vss_eev_run{seed}")

        try:
            results_eev = scsm.solve_model(
                model_eev, threads=args.threads, verbose=False,
                precision=args.precision, time_limit=args.time_limit,
                node_file_folder=node_file
            )

            eev_metrics = extract_vss_metrics(model_eev, "EEV")
            EEV = eev_metrics.get("objective")

        except Exception as e:
            print(f"  EEV solve failed (likely infeasible with fixed EV decisions): {e}")
            EEV = None
            eev_metrics = {"model_type": "EEV", "objective": None}

        print(f"  EEV objective: {EEV}")

        # ---- Step 4: Compute VSS ----
        if RP is not None and EEV is not None:
            VSS = RP - EEV
            VSS_pct = (VSS / abs(RP)) * 100 if RP != 0 else 0
        else:
            VSS = None
            VSS_pct = None

        print(f"\n  {'='*50}")
        print(f"  VALUE OF STOCHASTIC SOLUTION (Run {run_idx + 1})")
        print(f"  {'='*50}")
        print(f"  RP  (stochastic)  = {RP:,.2f}" if RP else "  RP  = N/A")
        print(f"  EV  (deterministic) = {EV:,.2f}" if EV else "  EV  = N/A")
        print(f"  EEV (EV decisions)  = {EEV:,.2f}" if EEV else "  EEV = N/A (infeasible)")
        print(f"  VSS = RP - EEV      = {VSS:,.2f}" if VSS else "  VSS = N/A")
        print(f"  VSS as % of RP      = {VSS_pct:.2f}%" if VSS_pct else "  VSS% = N/A")
        print(f"  EV_PI (perfect info) = {EV_PI:,.2f}" if EV_PI else "  EV_PI = N/A")
        print(f"  EVPI = EV_PI - RP    = {EVPI:,.2f}" if EVPI is not None else "  EVPI = N/A")

        if EEV is not None and EEV > RP:
            print(f"  WARNING: EEV > RP — this should not happen. Check model consistency.")

        # Record
        row = {
            "run": seed,
            "subsidy": args.subsidy,
            "deviation": args.deviation,
            "branches_s2": args.branches_stage2,
            "branches_s3": args.branches_stage3,
            "RP": RP,
            "EV": EV,
            "EEV": EEV,
            "EV_PI": EV_PI,
            "EVPI": EVPI,
            "VSS": VSS,
            "VSS_pct": VSS_pct,
            # RP details
            "rp_avg_h2": rp_metrics.get("avg_h2_production"),
            "rp_worst_scenario": rp_metrics.get("worst_scenario_obj"),
            "rp_stage1_entry": rp_metrics.get("total_stage1_entry"),
            # EEV details
            "eev_avg_h2": eev_metrics.get("avg_h2_production"),
            "eev_worst_scenario": eev_metrics.get("worst_scenario_obj"),
        }
        all_results.append(row)

    # Save all results
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")

    # Final summary
    if len(all_results) > 0:
        print("\n" + "=" * 70)
        print("VSS EXPERIMENT SUMMARY")
        print("=" * 70)
        vss_values = [r["VSS"] for r in all_results if r["VSS"] is not None]
        vss_pcts = [r["VSS_pct"] for r in all_results if r["VSS_pct"] is not None]
        if vss_values:
            print(f"  Across {len(vss_values)} runs:")
            print(f"    Mean VSS:    {np.mean(vss_values):,.2f} ({np.mean(vss_pcts):.2f}% of RP)")
            print(f"    Std VSS:     {np.std(vss_values):,.2f}")
            print(f"    Min VSS:     {np.min(vss_values):,.2f}")
            print(f"    Max VSS:     {np.max(vss_values):,.2f}")
            if np.mean(vss_pcts) > 5:
                print(f"  → Stochastic formulation provides SIGNIFICANT value over deterministic.")
            elif np.mean(vss_pcts) > 1:
                print(f"  → Stochastic formulation provides MODERATE value over deterministic.")
            else:
                print(f"  → Stochastic formulation provides MARGINAL value — consider if complexity is justified.")

        evpi_values = [r["EVPI"] for r in all_results if r.get("EVPI") is not None]
        if evpi_values:
            print(f"\n  EVPI summary across {len(evpi_values)} runs:")
            print(f"    Mean EVPI: {np.mean(evpi_values):,.2f}")
            print(f"    Std EVPI:  {np.std(evpi_values):,.2f}")
            print(f"    Min EVPI:  {np.min(evpi_values):,.2f}")
            print(f"    Max EVPI:  {np.max(evpi_values):,.2f}")

    print("\nFinished VSS experiment!", flush=True)
