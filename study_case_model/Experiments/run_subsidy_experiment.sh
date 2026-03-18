#!/bin/bash
# Set Job Requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=rome
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=l.m.j.beemster@student.vu.nl
#SBATCH --account=vusr121427    
#SBATCH --job-name=h2_stoch     
#SBATCH --output=logs/%j_out.txt 

# --- 1. Load Modules ---
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0  # Matches Gurobi's requirement
module load Gurobi/12.0.3-GCCcore-14.2.0

# --- 2. Activate Virtual Environment + License ---
export GRB_LICENSE_FILE="${HOME}/gurobi.lic"
source ${HOME}/hydrogen_venv/bin/activate

# --- 3. Dynamic Path Branching ---
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
UNIQUE_ID="${GIT_BRANCH}_job${SLURM_JOB_ID}"

# Define the NEW base directory in your HOME folder
# This will result in: /home/[user]/results/[branch]_job[id]/...
HOME_BASE="$HOME"

# Construct the full paths to match your script's structure but inside the unique folder
DATA_PATH="$HOME_BASE/study_case_model/scenario_variables/examine_subsidies/$UNIQUE_ID/"
FIG_PATH="$HOME_BASE/study_case_model/figures/examine_subsidies/$UNIQUE_ID/"

# Create the directories before running the Python script
mkdir -p "$DATA_PATH"
mkdir -p "$FIG_PATH"

echo "Running Branch: $GIT_BRANCH"
echo "Saving results directly to HOME: $HOME_BASE"

# --- 4. Execution ---
# Ensure the Python script is executable
chmod +x $HOME_BASE/projects/hydrogen_gas/study_case_model/Experiments/examine_subsidies.py

cd $HOME/projects/hydrogen_gas

# We override your Python defaults by passing these new paths as arguments
srun python study_case_model/Experiments/examine_subsidies.py \
    --amount_per_point 2 \
    --branches_stage2 2 \
    --branches_stage3 2 \
    --subsidies 0 40 80 \
    --deviations 0 0.1 \
    --data_folder "$DATA_PATH" \
    --figures_folder "$FIG_PATH"

echo "Experiment complete. Find your results in $HOME_BASE"