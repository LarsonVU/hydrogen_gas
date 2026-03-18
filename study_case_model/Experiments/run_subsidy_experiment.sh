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

module purge
module load 2025
module load Anaconda3/2025.06-1
module load Gurobi/12.0.3-GCCcore-14.2.0

export GRB_LICENSE_FILE="${HOME}/gurobi.lic"

# --- TMPDIR setup (template style) ---
mkdir -p "$TMPDIR"/data
mkdir -p "$TMPDIR"/figures

# --- Activate environment (same syntax as template) ---
source activate hydrogen_venv

# --- Ensure logs folder exists ---
mkdir -p logs

# --- Run experiment (NO loops) ---
python $HOME/projects/hydrogen_gas/study_case_model/Experiments/examine_subsidies.py \
    --amount_per_point 2 \
    --branches_stage2 2 \
    --branches_stage3 2 \
    --subsidies 0 40 80 \
    --deviations 0 0.1 \
    --data_folder "$TMPDIR/data/" \
    --figures_folder "$TMPDIR/figures/"

# --- Copy results back (template style) ---
mkdir -p $HOME/study_case_model/scenario_variables/examine_subsidies
mkdir -p $HOME/study_case_model/figures/examine_subsidies

cp -r "$TMPDIR"/data/* $HOME/study_case_model/scenario_variables/examine_subsidies/
cp -r "$TMPDIR"/figures/* $HOME/study_case_model/figures/examine_subsidies/

echo "Experiment complete. Results copied to HOME."