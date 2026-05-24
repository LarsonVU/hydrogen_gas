import os

# =========================
# Parameters (same as your script)
# =========================
# Different density of subsidies to illuminate trend
subsidies = [0,30, 45, 70]  # 0,5,...,80
deviations = [0, 0.05, 0.1, 0.2 ,1]
runs = 4

homogeneous_splits = [5, 10, 15, 20, 25]
density_bounds = [1, 2, 4, 8]

branches_stage2 = 4
branches_stage3 = 4

pickle_folder = "study_case_model/figures/speed_experiment_1a/run_24526/"
threads = 48
output_file = "study_case_model/Experiments/slurm_files/speed_jobs_24526.txt"
precision = 0.002

# =========================
# Generate jobs
# =========================
lines = []
for split in homogeneous_splits:
    for density in density_bounds:
        for dev_idx, dev in enumerate(deviations):
            for sub_idx, sub in enumerate(subsidies):
                cmd = (
                    "python study_case_model/Experiments/python_files/fixed_scenario_tree.py "
                    f"--runs {runs} "
                    f"--subsidy {sub} "
                    f"--deviation {dev} "
                    f"--upper_bounds 1 "
                    f"--precision {precision} "
                    f"--pickle_folder {pickle_folder} "
                    f"--threads {threads} "
                    f"--homogeneous_splits {split} "
                    f"--density_bounds {density} "
                    f"--branches_stage2 {branches_stage2} "
                    f"--branches_stage3 {branches_stage3} "
                )

                lines.append(cmd)

# =========================
# Write file
# =========================
with open(output_file, "w") as f:
    for line in lines:
        f.write(line + "\n")

print(f"Generated {len(lines)} jobs in {output_file}")