"""
Generate SLURM job commands for volatility stress test.
"""
import itertools

subsidies = [0, 30, 50]
vol_multipliers_str = "1.0,2.0,3.0,5.0,7.0,10.0"
correlated_options = [False, True]
runs = [0, 1, 2, 3]
date = "05052026"

with open("study_case_model/Experiments/slurm_files/volatility_stress_jobs.txt", "w") as f:
    for subsidy, correlated, run in itertools.product(subsidies, correlated_options, runs):
        corr_flag = " --correlated_shocks" if correlated else ""
        cmd = (
            f"python study_case_model/Experiments/python_files/examine_volatility_stress_test.py "
            f"--subsidy {subsidy} --run {run} "
            f"--vol_multipliers {vol_multipliers_str} "
            f"--branches_stage2 4 --branches_stage3 4 "
            f"--threads 48 --precision 0.01 --time_limit 600 "
            f"{corr_flag}"
            f"--output_csv study_case_model/figures/volatility_stress/{date}/stress_test_results_{subsidy}_{correlated}_{run}_.csv"
        )
        f.write(cmd + "\n")

print(f"Generated {len(subsidies) * len(correlated_options) * len(runs)} jobs")
