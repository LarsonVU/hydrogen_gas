"""
Extreme Volatility / Energy Crisis Stress Test
================================================

Tests how the optimal hydrogen blending strategy changes under
extreme price and demand volatility, simulating conditions like the
2022 European energy crisis.

Experiment design:
- Parametrically increase price and/or demand volatility from baseline
  (10-15% std) up to extreme levels (50-100% std).
- Optionally introduce correlated market shocks (all markets move together
  during a crisis, unlike normal independent fluctuations).
- Measure how the optimal objective, hydrogen production, and booking
  strategy change under stress.

This tests the robustness of the stochastic formulation: if the optimal
policy changes drastically under higher volatility, the current variance
assumptions are critical and should be well-justified.

Usage:
    python examine_volatility_stress_test.py --run 0 --subsidy 30
    python examine_volatility_stress_test.py --run 0 --subsidy 30 --correlated_shocks

Output:
    CSV with results across volatility levels.
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


# =========================
# Argument parsing
# =========================
parser = argparse.ArgumentParser(description="Energy crisis stress test for hydrogen gas network")

parser.add_argument("--run", type=int, default=0)
parser.add_argument("--branches_stage2", type=int, default=4)
parser.add_argument("--branches_stage3", type=int, default=4)
parser.add_argument("--subsidy", type=float, default=0, help="H2 subsidy in EUR/MWh")
parser.add_argument("--deviation", type=float, default=0.0)

# Volatility multipliers to test
parser.add_argument("--vol_multipliers", type=str, default="1.0,2.0,3.0,5.0,7.0,10.0",
                    help="Comma-separated volatility multipliers (1.0 = baseline)")
parser.add_argument("--correlated_shocks", action="store_true",
                    help="Enable correlated market shocks (crisis mode: all markets move together)")
parser.add_argument("--demand_shock_only", action="store_true",
                    help="Only increase demand volatility, keep price stable")
parser.add_argument("--price_shock_only", action="store_true",
                    help="Only increase price volatility, keep demand stable")

parser.add_argument("--upper_bounds", type=int, default=1)
parser.add_argument("--time_limit", type=float, default=600)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--precision", type=float, default=0.01)

parser.add_argument("--data_folder", type=str, default="scenario_variables/volatility_stress/")
parser.add_argument("--pickle_folder", type=str, default="study_case_model/figures/volatility_stress/")
parser.add_argument("--output_csv", type=str, default="study_case_model/figures/volatility_stress/stress_test_results.csv")

args = parser.parse_args()


# =========================
# Constants
# =========================
NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {1: 1, 2: args.branches_stage2, 3: args.branches_stage3}


# =========================
# Helper functions
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


def apply_volatility_multiplier(G, vol_multiplier, demand_mult=True, price_mult=True):
    """
    Scale the demand variance and price standard deviations by a multiplier.

    vol_multiplier = 1.0 means baseline.
    vol_multiplier = 3.0 means 3x baseline volatility (e.g., 30% demand variance -> 90%).
    vol_multiplier = 10.0 simulates extreme energy crisis conditions.
    """
    G_stressed = G.copy()

    for node in G_stressed.nodes:
        node_data = G_stressed.nodes[node]

        # Scale demand variance
        if demand_mult and "demand_variance" in node_data and not pd.isna(node_data["demand_variance"]):
            original_var = float(node_data["demand_variance"])
            G_stressed.nodes[node]["demand_variance"] = original_var * vol_multiplier

        # Scale long-term price std
        if price_mult and "long_term_price_std" in node_data and not pd.isna(node_data["long_term_price_std"]):
            original_std = float(node_data["long_term_price_std"])
            G_stressed.nodes[node]["long_term_price_std"] = original_std * vol_multiplier

        # Scale day-ahead price std
        if price_mult and "day_ahead_price_std" in node_data and not pd.isna(node_data["day_ahead_price_std"]):
            original_std = float(node_data["day_ahead_price_std"])
            G_stressed.nodes[node]["day_ahead_price_std"] = original_std * vol_multiplier

    return G_stressed


def add_correlated_demand_scenarios(scenarios, branches_per_stage, filename, correlation=0.8):
    """
    Modified demand scenario generation with correlated shocks across markets.

    In a crisis, demand shocks are correlated: when one market sees high demand,
    others tend to as well (e.g., cold snap across Europe). This replaces the
    independent sampling in the base model.

    correlation: fraction of demand shock that is common across all markets.
        0.0 = independent (baseline behavior)
        1.0 = perfectly correlated (all markets get same % shock)
    """
    demand_df = pd.DataFrame(columns=["stage", "scenario_index", "node", "supplier", "demand"])

    for scenario in scenarios[2]:
        # Common shock component (shared across all markets)
        common_shock = np.random.normal(0, 1)

        for node in scenario.G.nodes:
            node_data = scenario.G.nodes[node]
            if "average_demand_mwh_x1000" in node_data and not pd.isna(node_data["average_demand_mwh_x1000"]):
                avg_demand = float(node_data["average_demand_mwh_x1000"]) * 1000
                variance = float(node_data.get("demand_variance", 0))

                # Correlated + idiosyncratic components
                idio_shock = np.random.normal(0, 1)
                combined_shock = correlation * common_shock + np.sqrt(1 - correlation**2) * idio_shock
                variance_multiplier = 1 + variance * combined_shock
                sampled_demand = max(avg_demand * variance_multiplier, 0)

                if "supplier_ratios" in node_data and not pd.isna(node_data["supplier_ratios"]):
                    supplier_ratios = node_data["supplier_ratios"][0] if isinstance(node_data["supplier_ratios"], list) else node_data["supplier_ratios"]
                    scenario.G.nodes[node]["demand"] = {
                        supplier: sampled_demand * ratio
                        for supplier, ratio in supplier_ratios.items()
                    }
                    for supplier, demand_value in scenario.G.nodes[node]["demand"].items():
                        demand_df.loc[len(demand_df)] = [scenario.stage, scenario.index, node, supplier, demand_value]

    demand_df.to_excel(filename, index=False)

    # Stage 3 inherits from stage 2 predecessor (same as base model)
    for scenario in scenarios[3]:
        predecessor = scenario.predecessor
        for node in scenario.G.nodes:
            if node in predecessor.G.nodes and "demand" in predecessor.G.nodes[node]:
                scenario.G.nodes[node]["demand"] = predecessor.G.nodes[node]["demand"]


def create_scenarios_with_correlation(n_stages, b_stages, G, seed, folder, correlation=0.0):
    """
    Wrapper around scpf.create_scenarios that optionally adds correlated shocks.

    If correlation > 0, replaces demand scenarios with correlated version.
    """
    os.makedirs(folder, exist_ok=True)
    if seed is not None:
        np.random.seed(seed)

    scenarios = {k: [] for k in range(1, n_stages + 1)}
    stage_probs = scpf.prob_per_stage(n_stages, b_stages)

    branches = 1
    for k in range(1, n_stages + 1):
        branches = b_stages[k] * branches
        for m in range(1, branches + 1):
            scenario = scpf.Scenario(
                k, m, stage_probs[(k, m)], G.copy(),
                predecessor=scenarios[k-1][int((m-1) // b_stages[k])] if k > 1 else None
            )
            scenarios[k].append(scenario)

    # Use correlated demand if requested, otherwise standard
    if correlation > 0:
        add_correlated_demand_scenarios(
            scenarios, b_stages,
            filename=os.path.join(folder, f"demand_scenarios_corr{''.join(str(v) for v in b_stages.values())}.xlsx"),
            correlation=correlation
        )
    else:
        scpf.add_demand_scenarios(
            scenarios, branches_per_stage=b_stages,
            filename=os.path.join(folder, f"demand_scenarios{''.join(str(v) for v in b_stages.values())}.xlsx")
        )

    scpf.add_price_scenarios(
        scenarios, branches_per_stage=b_stages,
        filename=os.path.join(folder, f"price_scenarios{''.join(str(v) for v in b_stages.values())}.xlsx")
    )
    scpf.add_generation_costs(scenarios, branches_per_stage=b_stages)
    scpf.add_booking_costs(scenarios, branches_per_stage=b_stages)

    return scenarios


def extract_stress_metrics(model, results):
    """Extract metrics relevant to stress testing."""
    metrics = {}

    term_cond = str(results.solver.termination_condition)
    metrics["termination_condition"] = term_cond
    metrics["feasible"] = term_cond in ("optimal", "feasible")

    if not metrics["feasible"]:
        return {**metrics, "objective": None, "avg_h2_production": None,
                "avg_revenue": None, "scenario_obj_std": None,
                "worst_scenario_obj": None, "avg_demand_met": None}

    try:
        metrics["objective"] = pyo.value(model.objective)

        # H2 production stats
        h2_vals = [pyo.value(model.h2_production[m_3]) for m_3 in model.M[3]]
        metrics["avg_h2_production"] = np.mean(h2_vals)
        metrics["std_h2_production"] = np.std(h2_vals)

        # Scenario objective stats (key stress metric)
        obj_vals = [pyo.value(model.scenario_objective[m_3]) for m_3 in model.M[3]]
        metrics["avg_scenario_obj"] = np.mean(obj_vals)
        metrics["std_scenario_obj"] = np.std(obj_vals)
        metrics["worst_scenario_obj"] = np.min(obj_vals)
        metrics["best_scenario_obj"] = np.max(obj_vals)
        metrics["obj_range"] = np.max(obj_vals) - np.min(obj_vals)

        # Conditional Value at Risk (CVaR) approximation at 5th percentile
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

        # Booking strategy: total booking across stages
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

    except Exception as e:
        print(f"Warning: metrics extraction failed: {e}")

    return metrics


# =========================
# Main execution
# =========================
if __name__ == "__main__":
    vol_multipliers = [float(x) for x in args.vol_multipliers.split(",")]

    # Determine which volatilities to scale
    demand_mult = not args.price_shock_only
    price_mult = not args.demand_shock_only

    shock_type = "both"
    if args.demand_shock_only:
        shock_type = "demand_only"
    elif args.price_shock_only:
        shock_type = "price_only"

    print(f"Stress test configuration:")
    print(f"  Volatility multipliers: {vol_multipliers}")
    print(f"  Shock type: {shock_type}")
    print(f"  Correlated shocks: {args.correlated_shocks}")
    print(f"  Subsidy: {args.subsidy} EUR/MWh")
    print(f"  Deviation allowance: {args.deviation}")

    # Build base graph
    G_base = scpf.build_base_graph()
    subsidy_mscm = subsidy_per_mwh_to_mscm(args.subsidy)
    G_base = apply_subsidy(G_base, subsidy_mscm)

    all_results = []

    for vol_mult in vol_multipliers:
        print(f"\n{'='*60}")
        print(f"Volatility multiplier: {vol_mult}x")
        print(f"{'='*60}")

        # Apply volatility scaling
        G_stressed = apply_volatility_multiplier(
            G_base, vol_mult,
            demand_mult=demand_mult, price_mult=price_mult
        )

        # Create folders
        label = f"vol{vol_mult}_{'corr' if args.correlated_shocks else 'indep'}_{shock_type}"
        data_folder = os.path.join(args.data_folder, label, f"sub{args.subsidy}", f"run{args.run}")
        pickle_folder = os.path.join(args.pickle_folder, label, f"sub{args.subsidy}", f"run{args.run}")
        os.makedirs(data_folder, exist_ok=True)
        os.makedirs(pickle_folder, exist_ok=True)

        # Create scenarios
        correlation = 0.8 if args.correlated_shocks else 0.0
        scenarios = create_scenarios_with_correlation(
            NUMBER_OF_STAGES, BRANCHES_PER_STAGE, G_stressed,
            seed=args.run, folder=data_folder, correlation=correlation
        )

        # Build and solve model
        model = scsm.create_model(
            G_stressed, scenarios,
            allowed_deviation=args.deviation,
            number_of_density_bounds=args.upper_bounds
        )

        node_file_folder = os.environ.get("TMPDIR", "/tmp")
        node_file_folder = os.path.join(node_file_folder, f"gurobi_stress_{label}_run{args.run}")

        try:
            results = scsm.solve_model(
                model,
                threads=args.threads,
                verbose=False,
                precision=args.precision,
                time_limit=args.time_limit,
                node_file_folder=node_file_folder
            )

            metrics = extract_stress_metrics(model, results)

            if metrics["feasible"]:
                scsm.save_model_values(model, os.path.join(pickle_folder, "model_snapshot.pkl"))

        except Exception as e:
            print(f"  Solver error: {e}")
            metrics = {"termination_condition": "error", "feasible": False,
                       "objective": None}

        row = {
            "vol_multiplier": vol_mult,
            "shock_type": shock_type,
            "correlated": args.correlated_shocks,
            "subsidy": args.subsidy,
            "deviation": args.deviation,
            "run": args.run,
            **metrics
        }
        all_results.append(row)

        if metrics["feasible"]:
            print(f"  Objective: {metrics['objective']:.2f}")
            print(f"  Avg H2 production: {metrics.get('avg_h2_production', 'N/A')}")
            print(f"  Scenario obj std: {metrics.get('std_scenario_obj', 'N/A')}")
            print(f"  CVaR 5%: {metrics.get('cvar_5pct', 'N/A')}")
            print(f"  Worst scenario: {metrics.get('worst_scenario_obj', 'N/A')}")
        else:
            print(f"  INFEASIBLE ({metrics['termination_condition']})")

    # Save results
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")

    # Summary
    print("\n" + "=" * 80)
    print("STRESS TEST SUMMARY")
    print("=" * 80)
    baseline_obj = None
    for _, row in df.iterrows():
        if row["feasible"]:
            if row["vol_multiplier"] == 1.0:
                baseline_obj = row["objective"]
            loss = f"{((baseline_obj - row['objective']) / abs(baseline_obj) * 100):.1f}%" if baseline_obj else "N/A"
            print(f"  {row['vol_multiplier']:5.1f}x vol: obj={row['objective']:12.2f}, "
                  f"loss={loss}, "
                  f"H2={row.get('avg_h2_production', 0):8.2f}, "
                  f"CVaR5={row.get('cvar_5pct', 'N/A')}")
        else:
            print(f"  {row['vol_multiplier']:5.1f}x vol: INFEASIBLE")

    print("\nFinished stress test!", flush=True)
