import os
import numpy as np
# =========================
# Parameters (same as your script)
# =========================
subsidies = [30, 45, 70] 
allowed_hydrogen = np.linspace(0, 0.2, 21) 
runs = 4

data_folder = "study_case_model/scenario_variables/technical_experiment/run_07426/"
pickle_folder = "study_case_model/figures/technical_experiment/run_07426/"
threads = 48
output_file = "study_case_model/Experiments/slurm_files/technical_jobs.txt"
precision = 0.002

# =========================
# Generate jobs
# =========================
lines = []
for run_idx in range(runs):
    for sub_idx, sub in enumerate(subsidies):
        for al_h2 in allowed_hydrogen:
            cmd = (
                "python study_case_model/Experiments/python_files/examine_market_restrictions.py "
                f"--run {run_idx} "
                f"--branches_stage2 8 "
                f"--branches_stage3 8 "
                f"--subsidy {sub} "
                f"--allowed_hydrogen {al_h2} "
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