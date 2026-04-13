import os
import numpy as np
# =========================
# Parameters (same as your script)
# =========================
subsidies = [30, 45, 70] 
generation_plants = ["GJØA", "VISUND", "NORNE_ERB"]
pipelines = [("KÅRSTØ", "DRAUPNER_S"), ("KÅRSTØ", "DORNUM"), ("DRAUPNER_S", "DUNKERQUE"), ("H-7_BP", "EMDEN")]

runs = 4

data_folder = "study_case_model/scenario_variables/failure_experiment/run_13426_2/"
pickle_folder = "study_case_model/figures/failure_experiment/run_13426_2/"
threads = 48
output_file = "study_case_model/Experiments/slurm_files/failure_jobs.txt"
precision = 0.002

# =========================
# Generate jobs
# =========================
lines = []
for run_idx in range(runs):
    for sub_idx, sub in enumerate(subsidies):
        for failed_pipe in pipelines:
            cmd = (
                "python study_case_model/Experiments/python_files/examine_network_failure.py "
                f"--run {run_idx} "
                f"--branches_stage2 8 "
                f"--branches_stage3 8 "
                f"--subsidy {sub} "
                f"--failed_pipe_from {failed_pipe[0]}  "
                f"--failed_pipe_to {failed_pipe[1]} "
                f"--upper_bounds 1 "
                f"--precision {precision} "
                f"--data_folder {data_folder} "
                f"--pickle_folder {pickle_folder} "
                f"--threads {threads}"
            )

            lines.append(cmd)
        for failed_plant in generation_plants:
            cmd = (
                "python study_case_model/Experiments/python_files/examine_network_failure.py "
                f"--run {run_idx} "
                f"--branches_stage2 8 "
                f"--branches_stage3 8 "
                f"--subsidy {sub} "
                f"--failed_plant {failed_plant} "
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