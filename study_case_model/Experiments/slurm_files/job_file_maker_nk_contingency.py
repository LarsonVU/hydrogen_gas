"""
Generate SLURM job commands for N-k contingency analysis.
"""
import itertools

subsidies = [0, 30, 50]
max_k_values = [1, 2]
runs = [0, 1, 2, 3]

with open("nk_contingency_jobs.txt", "w") as f:
    for subsidy, max_k, run in itertools.product(subsidies, max_k_values, runs):
        cmd = (
            f"python study_case_model/Experiments/python_files/examine_nk_contingency.py "
            f"--subsidy {subsidy} --max_k {max_k} --run {run} "
            f"--branches_stage2 4 --branches_stage3 4 "
            f"--threads 48 --precision 0.01 --time_limit 600"
        )
        f.write(cmd + "\n")

print(f"Generated {len(subsidies) * len(max_k_values) * len(runs)} jobs")
