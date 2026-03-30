import os

# =========================
# Parameters (same as your script)
# =========================
subsidies = [0,25] + [30 + i * 2 for i in range(6)] + [45 + 5* i for i in range(5)] + [70]  # 0,5,...,80
deviations = [0, 0.05, 0.1, 0.2 ,1]
runs = 4

data_folder = "study_case_model/scenario_variables/subsidy_experiment/run_30326/"
pickle_folder = "study_case_model/figures/subsidy_experiment/run_30326/"
threads = 32
output_file = "study_case_model/Experiments/slurm_files/jobs_30326.txt"
precision = 0.002

# =========================
# Generate jobs
# =========================
lines = []
for run_idx in range(3,runs +3):
    for dev_idx, dev in enumerate(deviations):
        for sub_idx, sub in enumerate(subsidies):
            cmd = (
                "python study_case_model/Experiments/python_files/examine_parallel_subsidies.py "
                f"--run {run_idx} "
                f"--branches_stage2 8 "
                f"--branches_stage3 8 "
                f"--subsidy {sub} "
                f"--deviation {dev} "
                f"--upper_bounds 1 "
                f"--precision {precision}"
                f"--data_folder {data_folder} "
                f"--pickle_folder {pickle_folder} "
                f"--threads {threads}"
            )

            lines.append(cmd)

# =========================
# Write file
# =========================
with open(output_file, "w") as f:
    for line in lines:
        f.write(line + "\n")

print(f"Generated {len(lines)} jobs in {output_file}")