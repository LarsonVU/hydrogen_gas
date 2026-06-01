"""
Force Hydrogen Constraint Experiment
=====================================

Tests the impact of forcing specific amounts of hydrogen into the network
as a hard constraint (instead of using subsidy incentives).

This helps determine:
- What is the minimum objective cost if we MUST use certain H2 volumes?
- How does forced H2 compare economically to subsidy-driven H2?
- What are the network stress points when H2 is mandated?

Experiment design:
- For each hydrogen forcing level (e.g., 10%, 20%, 30%, ... of total production):
  - Add constraint: total H2 production in stage 3 >= target level
  - Solve model
  - Save model snapshot and results

Usage:
    python force_hydrogen.py --run 0 --h2_levels 0,10,20,30,50,75,100
    python force_hydrogen.py --run 0 --h2_levels 0,10,20,30,50,75,100 --subsidy 0

Output:
    CSV with results across hydrogen forcing levels.
    Pickle model snapshots per configuration.
"""

import numpy as np
import pyomo.environ as pyo
import networkx as nx
import pandas as pd
import sys
import os
import argparse
from pathlib import Path

# Add parent directory
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import study_case_stochastic_model as scsm
import study_case_problem_file as scpf
from Experiments.python_files.experiment_utils import subsidy_per_mwh_to_mscm, apply_subsidy, change_demand_constraint


# =========================
# Argument parsing
# =========================
parser = argparse.ArgumentParser(description="Force hydrogen constraint experiment for hydrogen gas network")

parser.add_argument("--run", type=int, default=0)
parser.add_argument("--branches_stage2", type=int, default=4)
parser.add_argument("--branches_stage3", type=int, default=1)
parser.add_argument("--subsidy", type=float, default=0, help="H2 subsidy in EUR/MWh (applied in addition to forcing)")
parser.add_argument("--deviation", type=float, default=0.0)

# Hydrogen forcing levels to test (percentages of total potential production)
parser.add_argument("--h2_levels", type=str, default="0,1.0,2.0,3.0,5.0,7.5,10",
                    help="Comma-separated hydrogen forcing levels (percent of max capacity)")

parser.add_argument("--upper_bounds", type=int, default=1)
parser.add_argument("--time_limit", type=float, default=None)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--precision", type=float, default=0.002)

parser.add_argument("--data_folder", type=str, default="scenario_variables/force_hydrogen/")
parser.add_argument("--pickle_folder", type=str, default="study_case_model/figures/force_hydrogen/")
parser.add_argument("--output_csv", type=str, default="study_case_model/figures/force_hydrogen/force_h2_results.csv")

args = parser.parse_args()


# =========================
# Constants
# =========================
NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {1: 1, 2: args.branches_stage2, 3: args.branches_stage3}

# CO2 Cost  
co2_cost_ng = 2.134 * 1000
co2_cost_green_h2 = 1.080 / 11.94 * 1000

conversion_ng = 39.8 / 3.6 * 1000
conversion_h2 = 12.7 / 3.6 * 1000

co2_savings_unit = 200.65 * (co2_cost_ng - co2_cost_green_h2) * (conversion_h2 / conversion_ng)


# =========================
# Helper functions
# =========================
def add_h2_forcing_constraint(model, h2_percentage):
    """
    Add a constraint that forces total hydrogen production to be at least
    h2_percentage% of the maximum possible production capacity.

    h2_percentage = 0 means no constraint (baseline).
    h2_percentage = 100 means force maximum H2 production.
    """
    if h2_percentage == 0:
        # No constraint
        return model

    perc = float(h2_percentage) / 100.0

    # Add an indexed constraint for each stage-3 scenario so the forcing
    # requirement applies individually per scenario.
    def h2_forcing_rule(m, m_3):
        # Maximum hydrogen that could be produced from all hydrogen-capable
        # nodes (using per-node capacity and component ratio for H2).
        max_h2_per_scenario = sum(m.G[n] * m.alpha[n, 'H2'] for n in m.N_hg)

        # Require hydrogen production in this scenario to be at least the
        # requested percent of the per-scenario maximum.
        return m.h2_production[m_3] >= perc * max_h2_per_scenario

    model.h2_forcing = pyo.Constraint(model.M[3], rule=h2_forcing_rule)
    return model


def extract_forcing_metrics(model, results, h2_percentage):
    """Extract metrics relevant to hydrogen forcing experiments."""
    metrics = {
        "h2_forcing_percent": h2_percentage
    }

    term_cond = str(results.solver.termination_condition)
    metrics["termination_condition"] = term_cond
    metrics["feasible"] = term_cond in ("optimal", "feasible")

    if not metrics["feasible"]:
        return {**metrics, "objective": None, "avg_h2_production": None,
                "total_h2_production": None, "avg_revenue": None,
                "avg_demand_met": None, "booking_cost": None}

    try:
        metrics["objective"] = pyo.value(model.objective)

        # H2 production stats
        h2_vals = [pyo.value(model.h2_production[m_3]) for m_3 in model.M[3]]
        metrics["avg_h2_production"] = np.mean(h2_vals)
        metrics["std_h2_production"] = np.std(h2_vals)

        # Scenario objective stats
        obj_vals = [pyo.value(model.scenario_objective[m_3]) for m_3 in model.M[3]]
        metrics["avg_scenario_obj"] = np.mean(obj_vals)
        metrics["std_scenario_obj"] = np.std(obj_vals)
        metrics["worst_scenario_obj"] = np.min(obj_vals)
        metrics["best_scenario_obj"] = np.max(obj_vals)

        # CVaR approximation at 5th percentile
        sorted_objs = np.sort(obj_vals)
        n_tail = max(1, int(0.05 * len(sorted_objs)))
        metrics["cvar_5pct"] = np.mean(sorted_objs[:n_tail])

        # Revenue stats
        rev_vals = [pyo.value(model.revenue_scenario[m_3]) for m_3 in model.M[3]]
        metrics["avg_revenue"] = np.mean(rev_vals)
        metrics["std_revenue"] = np.std(rev_vals)

        # Demand satisfaction
        demand_fracs = []
        for m_3 in model.M[3]:
            delivered = sum(
                pyo.value(model.gcv_c[c]) * pyo.value(model.f[a, c, m_3])
                for n in model.N_m for a in model.A_n_plus[n] for c in model.C
            )
            demanded = sum(
                pyo.value(model.D[h, n, m_3])
                for h in model.H for n in model.N_m
            )
            demand_fracs.append(delivered / demanded if demanded > 0 else 1.0)

        metrics["avg_demand_met"] = np.mean(demand_fracs)
        metrics["min_demand_met"] = np.min(demand_fracs)

        # Booking strategy costs
        booking_costs = sum(
            pyo.value(model.booking_cost[k, m_k])
            for k in model.K for m_k in model.M[k]
        )
        metrics["total_booking_cost"] = booking_costs

        # Total entry/exit booking
        total_entry = sum(pyo.value(model.x_entry[n, k, m_k])
                          for n in model.N for k in model.K for m_k in model.M[k])
        total_exit = sum(pyo.value(model.x_exit[n, k, m_k])
                         for n in model.N for k in model.K for m_k in model.M[k])
        metrics["total_entry_booking"] = total_entry
        metrics["total_exit_booking"] = total_exit

        # Quality deviations used
        metrics["n_quality_deviations"] = sum(
            pyo.value(model.delta[m_3]) for m_3 in model.M[3]
        )

        # Cost breakdown
        metrics["production_cost"] = sum(
            pyo.value(model.generation_cost[n, m_k]) * pyo.value(model.q[n, m_k])
            for n in model.N_hg for m_k in model.M[1]
        ) if hasattr(model, 'generation_cost') else None

    except Exception as e:
        print(f"Warning: metrics extraction failed: {e}")

    return metrics


# =========================
# Main execution
# =========================
if __name__ == "__main__":
    h2_levels = [float(x) for x in args.h2_levels.split(",")]

    print(f"Hydrogen forcing experiment configuration:")
    print(f"  Hydrogen forcing levels: {h2_levels}%")
    print(f"  Subsidy: {args.subsidy} EUR/MWh")
    print(f"  Deviation allowance: {args.deviation}")
    print(f"  Number of stages: {NUMBER_OF_STAGES}")
    print(f"  Branches per stage: {BRANCHES_PER_STAGE}")

    # Build base graph
    G_base = scpf.build_base_graph()
    
    # Apply subsidy if specified (in addition to forcing)
    if args.subsidy > 0:
        subsidy_mscm = subsidy_per_mwh_to_mscm(args.subsidy)
        G_base = apply_subsidy(G_base, subsidy_mscm)

    all_results = []

    for h2_level in h2_levels:
        print(f"\n{'='*60}")
        print(f"Hydrogen forcing level: {h2_level}%")
        print(f"{'='*60}")

        # Create folders for this hydrogen level
        label = f"h2force_{h2_level:.1f}pct"
        data_folder = os.path.join(args.data_folder, label, f"sub{args.subsidy}", f"run{args.run}")
        pickle_folder = os.path.join(args.pickle_folder, label, f"sub{args.subsidy}", f"run{args.run}")
        os.makedirs(data_folder, exist_ok=True)
        os.makedirs(pickle_folder, exist_ok=True)

        # Create scenarios (same for all h2_levels)
        scenarios = scpf.create_scenarios(
            NUMBER_OF_STAGES, BRANCHES_PER_STAGE, G_base,
            seed=args.run, folder=data_folder
        )

        # Build model
        model = scsm.create_model(
            G_base, scenarios,
            allowed_deviation=args.deviation,
            number_of_density_bounds=args.upper_bounds
        )

        model = change_demand_constraint(model)

        # Add hydrogen forcing constraint
        if h2_level > 0:
            model = add_h2_forcing_constraint(model, h2_level)
            print(f"  Added H2 forcing constraint: >= {h2_level}% of max capacity")

        # Solve model
        node_file_folder = os.environ.get("TMPDIR", "/tmp")
        node_file_folder = os.path.join(node_file_folder, f"gurobi_force_h2_{label}_run{args.run}")

        try:
            results = scsm.solve_model(
                model,
                threads=args.threads,
                verbose=True,
                precision=args.precision,
                time_limit=args.time_limit,
                node_file_folder=node_file_folder
            )

            metrics = extract_forcing_metrics(model, results, h2_level)

            # Save model snapshot if feasible
            if metrics["feasible"]:
                model_snapshot_path = os.path.join(pickle_folder, "model_snapshot.pkl")
                scsm.save_model_values(model, model_snapshot_path)
                print(f"  Model saved to: {model_snapshot_path}")

        except Exception as e:
            print(f"  Solver error: {e}")
            metrics = {"h2_forcing_percent": h2_level, "termination_condition": "error", 
                       "feasible": False, "objective": None}

        row = {
            "h2_forcing_percent": h2_level,
            "subsidy": args.subsidy,
            "deviation": args.deviation,
            "run": args.run,
            **metrics
        }
        all_results.append(row)

        if metrics["feasible"]:
            print(f"  Objective: {metrics['objective']:.2f}")
            print(f"  Avg H2 per scenario: {metrics['avg_h2_production']:.2f}")
            print(f"  Avg revenue: {metrics.get('avg_revenue', None):.2f}")
            print(f"  Avg demand met: {metrics.get('avg_demand_met', None):.2%}")
        else:
            print(f"  INFEASIBLE ({metrics['termination_condition']})")

    # Compute net welfare effect relative to the unconstrained baseline
    baseline_obj = next(
        (row["objective"] for row in all_results
         if row["h2_forcing_percent"] == 0 and row["feasible"]),
        None
    )
    for row in all_results:
        if baseline_obj is None or not row.get("feasible", False):
            row["objective_change"] = None
            row["objective_pct_change"] = None
            row["net_welfare_change"] = None
            row["net_welfare_pct_change"] = None
        else:
            row["objective_change"] = row["objective"] - baseline_obj
            row["objective_pct_change"] = (
                (row["objective"] - baseline_obj) / abs(baseline_obj) * 100
                if baseline_obj != 0 else None
            )
            row["net_welfare_change"] = row["objective_change"] + co2_savings_unit * row.get("avg_h2_production", 0)
            row["net_welfare_pct_change"] = (
                (row["net_welfare_change"] / abs(baseline_obj) * 100) if baseline_obj != 0 else None
            )

    # Save results
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")

    # Summary
    print("\n" + "=" * 80)
    print("HYDROGEN FORCING EXPERIMENT SUMMARY")
    print("=" * 80)
    for row in all_results:
        if row["feasible"]:
            objective_pct_change = (
                f"{row['objective_pct_change']:+.1f}%" if row['objective_pct_change'] is not None else "N/A"
            )
            objective_change = (
                f"{row['objective_change']:+.2f}" if row['objective_change'] is not None else "N/A"
            )
            print(
                f"  {row['h2_forcing_percent']:6.1f}% forcing: obj={row['objective']:12.2f}, "
                f"objective_change={objective_change}, "
                f"objective_pct_change={objective_pct_change}, "
                f"H2={row.get('avg_h2_production', 0):10.2f}, "
                f"demand_met={row.get('avg_demand_met', 0):.1%}"
            )
            print(
                f"      net_welfare_change={row.get('net_welfare_change', 0):+12.2f}, "
                f"net_welfare_pct_change={row.get('net_welfare_pct_change', 0):+.1f}%"
            )
            print()
        else:
            print(f"  {row['h2_forcing_percent']:6.1f}% forcing: INFEASIBLE")

    print("\nFinished hydrogen forcing experiment!", flush=True)
