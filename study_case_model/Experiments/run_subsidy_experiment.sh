#!/bin/bash
# Set Job Requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=rome
#SBATCH --mail-type=BEGIN,END
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
source activate hydrogen_venv

# --- Verification Check ---
echo "--- Environment Check ---"
which python3               # Should point to ~/hydrogen_venv/bin/python3
python3 --version           # Should be 3.13.5
pip list | grep -E "numpy|pyomo|gurobipy" # Verify your key libraries are present
echo "--------------------------"

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
# We override your Python defaults by passing these new paths as arguments
srun python3 your_script_name.py \
    --amount_per_point 2 \
    --branches_stage2 2 \
    --branches_stage3 2 \
    --subsidies 0 40 80 \
    --deviations 0 0.1 \
    --data_folder "$DATA_PATH" \
    --figures_folder "$FIG_PATH"

echo "Experiment complete. Find your results in $HOME_BASE"