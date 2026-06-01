import os

# =========================
# Parameters (same as your script)
# =========================
# Different density of subsidies to illuminate trend
deviations = [0, 0.05, 0.1, 0.2 ,1]
runs = 4

data_folder = "study_case_model/scenario_variables/force_hydrogen/run_1626/"
pickle_folder = "study_case_model/figures/force_hydrogen/run_1626/"
csv_folder = "study_case_model/figures/force_hydrogen/run_1626/"
threads = 48
output_file = "study_case_model/Experiments/slurm_files/force_jobs_1626.txt"
precision = 0.002
h2_levels = ",".join(f"{i}" for i in range(1, 21))

# =========================
# Generate jobs
# =========================
lines = []
for run_idx in range(0,runs):
    for dev_idx, dev in enumerate(deviations):
            cmd = (
                "python study_case_model/Experiments/python_files/fixed_scenario_tree.py "
                f"--run {run_idx} "
                f"--h2_levels {h2_levels} "
                f"--deviation {dev} "
                f"--branches_stage2 8 "
                f"--branches_stage3 8 "
                f"--upper_bounds 1 "
                f"--precision {precision} "
                f"--h2_levels {h2_levels} "
                f"--data_folder {data_folder} "
                f"--pickle_folder {pickle_folder} "
                f"--output_csv {csv_folder + f"/dev{dev}/run_{run_idx}/" + 'force_h2_results.csv'} "
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