#!/bin/bash
# Set Job Requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=rome
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=l.m.j.beemster@student.vu.nl
#SBATCH --account=vusr121427    
#SBATCH --job-name=speed_stoch     
#SBATCH --output=logs/%j_out.txt 
#SBATCH --time=08:00:00

# --- 1. Load Modules ---
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0  # Matches Gurobi's requirement
module load Gurobi/12.0.3-GCCcore-14.2.0

# --- 2. Activate Virtual Environment + License ---
export GRB_LICENSE_FILE="${HOME}/gurobi.lic"
source activate hydrogen_venv

# --- 3. Dynamic Path Branching ---
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
UNIQUE_ID="${GIT_BRANCH}_job${SLURM_JOB_ID}"

# Define the NEW base directory in your HOME folder
# This will result in: /home/[user]/results/[branch]_job[id]/...
HOME_BASE="$HOME"

# Construct the full paths to match your script's structure but inside the unique folder
FIG_PATH="$HOME_BASE/projects/hydrogen_gas/study_case_model/figures/examine_speed/$UNIQUE_ID/"

# Create the directories before running the Python script
mkdir -p "$DATA_PATH"
mkdir -p "$FIG_PATH"

echo "Running Branch: $GIT_BRANCH"
echo "Saving results directly to HOME: $HOME_BASE"

# --- 4. Execution ---
# Ensure the Python script is executable
chmod +x $HOME_BASE/projects/hydrogen_gas/study_case_model/Experiments/python_files/examine_speed.py

cd $HOME/projects/hydrogen_gas

# We override your Python defaults by passing these new paths as arguments
srun python study_case_model/Experiments/python_files/examine_speed.py \
    --folder "$FIG_PATH" \
    --branches_stage1 1 \
    --branches_stage2 4 \
    --branches_stage3 4 \
    --precision 0.001
echo "Experiment complete. Find your results in $HOME_BASE"