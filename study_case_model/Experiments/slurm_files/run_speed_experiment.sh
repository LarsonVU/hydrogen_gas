#!/bin/bash
# Set Job Requirements
#SBATCH -t 10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=genoa
#SBATCH --array=1-1
#SBATCH --output=logs/%A_%a.out
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=l.m.j.beemster@student.vu.nl

# Loading modules
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load Gurobi/12.0.3-GCCcore-14.2.0

# =========================
# 2. Environment
# =========================
export GRB_LICENSE_FILE="${HOME}/gurobi.lic"

# ✅ FIXED venv activation
source activate hydrogen_venv

UNIQUE_ID="${GIT_BRANCH}_job${SLURM_JOB_ID}"
# Construct the full paths to match your script's structure but inside the unique folder
FIG_PATH="$HOME/projects/hydrogen_gas/study_case_model/figures/examine_speed/$UNIQUE_ID/"

cd $HOME/projects/hydrogen_gas

i=${SLURM_ARRAY_TASK_ID}
$(head -$i study_case_model/Experiments/slurm_files/speed_jobs.txt | tail -1)