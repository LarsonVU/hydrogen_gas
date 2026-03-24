import argparse
import numpy as np

from study_case_stochastic_model import (
    build_base_graph,
    create_scenarios,
    create_model,
    solve_model,
    generate_cutting_plane_pairs,
    save_model_values
)

# =========================
# Argument parsing
# =========================
parser = argparse.ArgumentParser()

parser.add_argument("--stages", type=int, default=3)
parser.add_argument("--branches_stage2", type=int, default=4)
parser.add_argument("--branches_stage3", type=int, default=4)
parser.add_argument("--deviation", type=float, default=0)

parser.add_argument("--rho_low", type=float, default=0.55)
parser.add_argument("--rho_high", type=float, default=0.70)

parser.add_argument("--splits", type=int, default=11)

parser.add_argument("--time_limit", type=float, default=None)

# NEW: output file argument
parser.add_argument("--output", type=str, default="test_run.pkl")

args = parser.parse_args()

# =========================
# Model parameters
# =========================
FOLDER = "study_case_model/figures/test_run/"

NUMBER_OF_STAGES = args.stages
BRANCHES_PER_STAGE = {
    1: 1,
    2: args.branches_stage2,
    3: args.branches_stage3
}

ALLOWED_DEVIATION = args.deviation

NUMBER_OF_DENSITY_BOUNDS = 1
RHO_LOW = args.rho_low
RHO_HIGH = args.rho_high

NUMBER_OF_HOMOGENEOUS_SPLITS = args.splits
splits_per_arc = np.linspace(0, 1, NUMBER_OF_HOMOGENEOUS_SPLITS)

# =========================
# Build and solve model
# =========================
G = build_base_graph()
scenarios = create_scenarios(NUMBER_OF_STAGES, BRANCHES_PER_STAGE, G)

model = create_model(
    G,
    scenarios,
    cutting_plane_pairs=generate_cutting_plane_pairs(method="skewed")
)

results = solve_model(model, time_limit=args.time_limit)
print(results)

# =========================
# Save results (uses argument)
# =========================
save_model_values(
    model,
    f"study_case_model/scenario_variables/{args.output}"
)