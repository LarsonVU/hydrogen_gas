"""
Generate SLURM job commands for volatility stress test.
"""
import itertools

subsidies = [0, 30, 45, 70]
vol_multipliers = "1.0,2.0,4.0,8.0,16.0"
correlations = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
shocks = ["price_shock_lt", "price_shock_st"]
runs = [0, 1, 2, 3]
date = "07052026"

with open("study_case_model/Experiments/slurm_files/volatility_stress_jobs.txt", "w") as f:
    for subsidy, correlation, shock, run in itertools.product(subsidies, correlations, shocks, runs):
        shock_flag = f" --{shock}" if shock else ""
        cmd = (
            f"python study_case_model/Experiments/python_files/examine_volatility_stress_test.py "
            f"--subsidy {subsidy} --run {run} "
            f"--vol_multipliers {vol_multipliers} "
            f"--branches_stage2 8 --branches_stage3 8 "
            f"--threads 48 --precision 0.002 "
            f"--correlation_price {correlation} --correlation_demand 0 "
            f"{shock_flag} {True} "
            f"--pickel_folder study_case_model/figures/volatility_stress/{date} "
            f"--output_csv study_case_model/figures/volatility_stress/{date}/stress_test_results_{subsidy}_{correlation}_{shock}_{run}_.csv"
        )
        f.write(cmd + "\n")

print(f"Generated {len(subsidies) * len(correlations) * len(shocks) * len(runs)} jobs")
