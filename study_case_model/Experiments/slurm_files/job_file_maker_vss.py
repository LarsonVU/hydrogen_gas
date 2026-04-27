"""
Generate SLURM job commands for Value of Stochastic Solution experiment.
"""
import itertools

subsidies = [0, 25, 30, 40, 50, 60]
deviations = [0, 0.05, 0.1]
runs = [0, 1, 2, 3]

with open("vss_jobs.txt", "w") as f:
    for subsidy, deviation, run in itertools.product(subsidies, deviations, runs):
        cmd = (
            f"python study_case_model/Experiments/python_files/examine_vss.py "
            f"--subsidy {subsidy} --deviation {deviation} --run {run} "
            f"--branches_stage2 4 --branches_stage3 4 "
            f"--threads 48 --precision 0.001 --time_limit 1200"
        )
        f.write(cmd + "\n")

print(f"Generated {len(subsidies) * len(deviations) * len(runs)} jobs")
