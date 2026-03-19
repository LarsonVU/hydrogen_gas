#!/bin/bash
#SBATCH --job-name=h2_array
#SBATCH --account=vusr121427
#SBATCH --partition=rome

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8

#SBATCH --time=02:00:00

# 🔥 204 tasks total, max 40 running at once
#SBATCH --array=0-203%40

#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

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
cd $HOME/projects/hydrogen_gas

DATA_PATH="$HOME/projects/hydrogen_gas/study_case_model/scenario_variables/examine_subsidies"

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# =========================
# 4. Run experiment
# =========================
srun python -u study_case_model/Experiments/examine_subsidies.py \
    --amount_per_point 4 \
    --branches_stage2 8 \
    --branches_stage3 8 \
    --subsidies $(seq 0 5 80) \
    --deviations 0 0.05 0.1 \
    --upper_bounds 4 \
    --data_folder "$DATA_PATH"