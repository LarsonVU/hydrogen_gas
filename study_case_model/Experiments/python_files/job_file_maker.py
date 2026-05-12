import os

# =========================
# Parameters (same as your script)
# =========================
# Different density of subsidies to illuminate trend
subsidies = [0,25] + [26 + i * 2 for i in range(8)] + [45 + 5* i for i in range(5)] + [70]  # 0,5,...,80
deviations = [0, 0.05, 0.1, 0.2 ,1]
runs = 4

data_folder = "study_case_model/scenario_variables/subsidy_experiment/run_12526/"
pickle_folder = "study_case_model/figures/subsidy_experiment/run_12526/"
threads = 48
output_file = "study_case_model/Experiments/slurm_files/jobs_12526.txt"
precision = 0.002

# =========================
# Generate jobs
# =========================
lines = []
for run_idx in range(0,runs):
    for dev_idx, dev in enumerate(deviations):
        for sub_idx, sub in enumerate(subsidies):
            cmd = (
                "python study_case_model/Experiments/python_files/fixed_scenario_tree.py "
                f"--run {run_idx} "
                f"--subsidy {sub} "
                f"--deviation {dev} "
                f"--upper_bounds 1 "
                f"--precision {precision} "
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