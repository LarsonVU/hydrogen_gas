#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=genoa
#SBATCH --array=1-10

#SBATCH --output=logs/%A_%a.out

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=l.m.j.beemster@student.vu.nl


# =========================
# 1. Modules
# =========================
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load Gurobi/12.0.3-GCCcore-14.2.0

# =========================
# 2. Environment
# =========================
export GRB_LICENSE_FILE="${HOME}/gurobi.lic"

# ✅ FIXED venv activation
source activate hydrogen_venv

# =========================
# 3. Paths
# =========================
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
UNIQUE_ID="${GIT_BRANCH}_job${SLURM_JOB_ID}"

DATA_PATH="$HOME/projects/hydrogen_gas/study_case_model/scenario_variables/parallel_subsidies"

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# =========================
# 4. Run experiment
# =========================
chmod +x $HOME/projects/hydrogen_gas/study_case_model/study_case_model/Experiments/python_files/examine_parallel_subsidies.py

cd $HOME/projects/hydrogen_gas

srun python study_case_model/Experiments/python_files/examine_parallel_subsidies.py \
    --amount_per_point 4 \
    --branches_stage2 8 \
    --branches_stage3 8 \
    --subsidies $(seq 0 5 80) \
    --deviations 0 0.05 0.1 \
    --upper_bounds 4 \
    --data_folder "$DATA_PATH"

echo "Experiment complete. Find your results in the folder"