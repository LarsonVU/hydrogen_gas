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
from Experiments.python_files.experiment_utils import subsidy_per_mwh_to_mscm, apply_subsidy, apply_technical_restriction

# =========================
# Argument parsing
# =========================
parser = argparse.ArgumentParser()

parser.add_argument("--run", type=int, default= 0 )
parser.add_argument("--branches_stage2", type=int, default=4)
parser.add_argument("--branches_stage3", type=int, default=4)
parser.add_argument("--subsidy", type=float, default=0)
parser.add_argument("--failed_pipe_from", type=str, default=None)
parser.add_argument("--failed_pipe_to", type=str, default=None)
parser.add_argument("--failed_plant", type=str, default= None)
parser.add_argument("--penalty", type=float, default=1000)

parser.add_argument("--upper_bounds", type=int, default=1)
parser.add_argument("--time_limit", type=float, default=None)
parser.add_argument("--threads", type= int, default= 8)
parser.add_argument("--precision", type=float, default=0.001)

parser.add_argument("--data_folder", type=str, default="scenario_variables/other_experiments/")
parser.add_argument("--pickle_folder", type= str, default= "study_case_model/figures/other_experiments")

args = parser.parse_args()


# =========================
# Constants
# =========================
NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {
    1: 1,
    2: args.branches_stage2,
    3: args.branches_stage3
}
UPPER_BOUNDS = args.upper_bounds
THREADS = args.threads
SUBSIDY = args.subsidy
RUN = args.run
DEMAND_PENALTY = args.penalty

FAILED = args.failed_pipe_from + "_to_" + args.failed_pipe_to if args.failed_pipe_from is not None else args.failed_plant

# =========================
# Helper functions
# =========================

def change_demand_constraint(model, missed_demand_penalty=1000):
    model.demand_satisfaction_production.deactivate()
    model.demand_satisfaction_market.deactivate()
    
    model.shortage_production = pyo.Var(model.H, model.M[3], domain=pyo.NonNegativeReals)
    model.shortage_market = pyo.Var(model.N_m, model.M[3], domain=pyo.NonNegativeReals)

    def soft_demand_satisfaction_production_rule(m, h, m_3):
        nodes_for_h = [n for n in m.N_hg if m.supplier[n] == h]

        if not nodes_for_h:
            return pyo.Constraint.Skip

        lhs = sum(
            m.gcv_c[c] * m.f[a, c, m_3]
            for c in m.C
            for n in nodes_for_h
            for a in m.A_n_minus[n]
        ) - sum(
            m.gcv_c[c] * m.f[a, c, m_3]
            for c in m.C
            for n in nodes_for_h
            for a in m.A_n_plus[n]
        )

        rhs = sum(m.D[h, n, m_3] for n in m.N_m)

        return lhs + m.shortage_production[h, m_3] >= rhs

    model.soft_demand_satisfaction_production = pyo.Constraint(
        model.H, model.M[3], rule=soft_demand_satisfaction_production_rule
    )   

    def soft_demand_satisfaction_market_rule(m, n, m_3):
        if n not in m.N_m:
            return pyo.Constraint.Skip

        lhs = sum(m.gcv_c[c] * m.f[a, c, m_3] for a in m.A_n_plus[n] for c in m.C)
        rhs = sum(m.D[h, n, m_3] for h in m.H)

        return lhs + m.shortage_market[n, m_3] >= rhs

    model.soft_demand_satisfaction_market = pyo.Constraint(
        model.N_m, model.M[3], rule=soft_demand_satisfaction_market_rule
    )

    model.penalty_production = pyo.Param(initialize=missed_demand_penalty, mutable=True)
    model.penalty_market = pyo.Param(initialize=missed_demand_penalty, mutable=True)

    # Store original objective expression and sense
    original_expr = model.objective.expr
    sense = model.objective.sense

    # Deactivate old objective
    model.del_component(model.objective)

    # Define penalty expression once (cleaner and reusable)
    def penalty_expression(m):
        return sum(
            m.penalty_production * m.shortage_production[h, m_3]
            for h in m.H for m_3 in m.M[3]
        ) + sum(
            m.penalty_market * m.shortage_market[n, m_3]
            for n in m.N_m for m_3 in m.M[3]
        )

    model.penalty_expression = pyo.Expression(rule=penalty_expression)
    model.base_objective = original_expr

    # New objective
    def objective_rule(m):
        return original_expr - penalty_expression(m)

    model.objective = pyo.Objective(rule=objective_rule, sense=sense)
    return model


# =========================
# Main execution
# =========================
if __name__ == "__main__":

    # Build base graph
    G = scpf.build_base_graph()

    # Convert subsidy
    subsidy_mscm = subsidy_per_mwh_to_mscm(SUBSIDY)

    # Apply subsidy
    G_changed = apply_subsidy(G, subsidy_mscm)
    # Apply market restriction
    G_changed = apply_technical_restriction(G_changed, failed_pipe_from=args.failed_pipe_from, failed_pipe_to=args.failed_pipe_to, failed_plant=args.failed_plant)

    # Create scenario folder
    data_folder = os.path.join(
        args.data_folder,
        f"maxh2_{FAILED}",
        f"sub{SUBSIDY}",
        f"run{RUN}"
    )

    # Create pickle folder
    pickle_folder = os.path.join(
        args.pickle_folder,
        f"maxh2_{FAILED}",
        f"sub{SUBSIDY}",
        f"run{RUN}"
    )

    os.makedirs(data_folder, exist_ok=True)

    # Create scenarios
    scenarios = scpf.create_scenarios(
        NUMBER_OF_STAGES,
        BRANCHES_PER_STAGE,
        G_changed,
        seed= RUN,
        folder=data_folder
    )

    print("Solving model:" + f", failed_{FAILED}, sub{SUBSIDY}, run{RUN}", flush=True)

    # Build model
    model = scsm.create_model(
        G_changed,
        scenarios,
        number_of_density_bounds=UPPER_BOUNDS
    )
    # Change demand constraints to allow for unmet demand with penalty
    model = change_demand_constraint(model, missed_demand_penalty=DEMAND_PENALTY)

    # Solve with Gurobi (multithreaded)
    node_file_folder = os.environ.get("TMPDIR", "/tmp")
    node_file_folder = os.path.join(node_file_folder, f"gurobi_failed_{FAILED}_sub{SUBSIDY}_run{RUN}")
    results = scsm.solve_model(model, threads= THREADS, verbose= True, precision=args.precision, node_file_folder=node_file_folder)

    # Save results
    scsm.save_model_values(model, os.path.join(pickle_folder, "model_snapshot.pkl"))
    print(f"Model with original objective value {pyo.value(model.base_objective)} and penalty {pyo.value(model.penalty_expression)}", flush=True)
    print("Finished successfully!", flush=True)