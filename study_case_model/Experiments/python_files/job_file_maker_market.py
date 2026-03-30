import os

# =========================
# Parameters (same as your script)
# =========================
subsidies = [0,25] + [30 + i * 2 for i in range(6)] + [45 + 5* i for i in range(5)] + [70]  # 0,5,...,80
deviations = [0, 0.05, 0.1, 0.2 ,1]
runs = 2

data_folder = "study_case_model/scenario_variables/market_experiment/run_30326/"
pickle_folder = "study_case_model/figures/market_experiment/run_30326/"
threads = 32
output_file = "study_case_model/Experiments/slurm_files/market_jobs.txt"

# =========================
# Generate jobs
# =========================
lines = []
for run_idx in range(5,runs +5):
    for dev_idx, dev in enumerate(deviations):
        for sub_idx, sub in enumerate(subsidies):
            cmd = (
                "python study_case_model/Experiments/python_files/examine_market_restrictions.py "
                f"--run {run_idx} "
                f"--branches_stage2 8 "
                f"--branches_stage3 8 "
                f"--subsidy {sub} "
                f"--deviation {dev} "
                f"--upper_bounds 1 "
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