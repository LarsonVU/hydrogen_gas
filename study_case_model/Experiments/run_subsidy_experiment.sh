#!/bin/bash
#SBATCH -p genoa                # Partition for the newest CPU nodes
#SBATCH -n 1                    # Number of tasks
#SBATCH -c 16                   # Number of CPU cores
#SBATCH --mem=32G               # Explicitly set memory (avoids "24 CPU charge" error)
#SBATCH -t 04:00:00             # Max time (HH:MM:SS)
#SBATCH --account=vusr121427    # <--- UPDATED: This is your Account ID from accinfo
#SBATCH --job-name=h2_stoch     
#SBATCH --output=logs/%j_out.txt 

# --- 1. Load Modules ---
module purge
module load 2025
module load Python/3.13.5-GCCcore-14.3.0
module load Gurobi/12.0.3-GCCcore-14.2.0 

# --- 2. Activate Virtual Environment ---
source ~/hydrogen_venv/bin/activate

# --- 3. Dynamic Path Branching ---
# Ensures we don't overwrite different git branch results
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
UNIQUE_ID="${GIT_BRANCH}_job${SLURM_JOB_ID}"

# Define the paths (relative to your hydrogen_gas directory)
DATA_PATH="study_case_model/scenario_variables/examine_subsidies/${UNIQUE_ID}/"
FIG_PATH="study_case_model/figures/examine_subsidies/${UNIQUE_ID}/"

mkdir -p "$DATA_PATH"
mkdir -p "$FIG_PATH"

echo "Running Branch: $GIT_BRANCH"
echo "Results will be saved in: $DATA_PATH"

# --- 4. Execution ---
# Note: Using the full path to the python script
srun python3 study_case_model/Experiments/examine_subsidies.py \
    --amount_per_point 2 \
    --branches_stage2 2 \
    --branches_stage3 2 \
    --subsidies 0 40 80 \
    --deviations 0 0.1 \
    --data_folder "$DATA_PATH" \
    --figures_folder "$FIG_PATH"

echo "Experiment complete."