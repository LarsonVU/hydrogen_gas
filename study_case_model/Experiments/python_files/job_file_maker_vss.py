"""
Generate SLURM job commands for Value of Stochastic Solution experiment.
"""
import itertools

subsidies = [0, 30, 45, 70]  # Subsidy levels to test
deviations = [0, 0.05, 0.1, 0.2, 1.0]  # Deviation levels to test
run = 0  # Starting run index (can be incremented for multiple runs)
branches_stage2 = [2, 4,8]
branches_stage3 = [2, 4,8]
n_runs = 4 # Number of independent runs for each scenario (with different seeds)
with open("study_case_model/Experiments/slurm_files/vss_jobs.txt", "w") as f:
    for subsidy, deviation, branches2, branches3, run in itertools.product(subsidies, deviations, branches_stage2, branches_stage3, range(n_runs)):
        cmd = (
            f"python study_case_model/Experiments/python_files/examine_vss.py "
            f"--subsidy {subsidy} --deviation {deviation} --run {run} "
            f"--branches_stage2 {branches2} --branches_stage3 {branches3} "
            f"--threads 48 --precision 0.002 "
            f"--data_folder study_case_model/scenario_variables/vss_experiment/28426/ "
            f"--pickle_folder study_case_model/figures/vss_experiment/28426/ "
            f"--output_csv study_case_model/figures/vss_experiment/28426/vss_results_{subsidy}_{deviation}_{branches2}_{branches3}.csv "
        )
        f.write(cmd + "\n")

print(f"Generated {len(subsidies) * len(deviations) * len(branches_stage2) * len(branches_stage3)} jobs")
