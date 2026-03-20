import os

# =========================
# Parameters (same as your script)
# =========================
subsidies = list(range(0, 81, 5))   # 0,5,...,80
deviations = [0, 0.05, 0.1]
runs = 4

data_folder = "study_case_model/scenario_variables/subsidy_experiment/"
pickle_folder = "study_case_model/figures/subsidy_experiment/"

output_file = "study_case_model/Experiments/slurm_files/jobs.txt"

# =========================
# Generate jobs
# =========================
lines = []

for dev_idx, dev in enumerate(deviations):
    for sub_idx, sub in enumerate(subsidies):
        for run_idx in range(runs):

            cmd = (
                "python study_case_model/Experiments/python_files/examine_parallel_subsidies.py "
                f"--run {run_idx} "
                f"--branches_stage2 8 "
                f"--branches_stage3 8 "
                f"--subsidies {sub} "
                f"--deviations {dev} "
                f"--upper_bounds 5 "
                f"--data_folder {data_folder} "
                f"--data_folder {pickle_folder} "
            )

            lines.append(cmd)

# =========================
# Write file
# =========================
with open(output_file, "w") as f:
    for line in lines:
        f.write(line + "\n")

print(f"Generated {len(lines)} jobs in {output_file}")