#!/bin/bash
# Set Job Requirements
#SBATCH -t 10:00:00
#SBATCH --nodes=1
#SBATCH -n 96
#SBATCH --partition=genoa
#SBATCH --array=1-5
#SBATCH --out=slurm/slurm-%A-%a.out
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=b.t.markhorst@student.vu.nl

# Loading modules
module load 2023
module load Julia/1.10.4-linux-x86_64
module load Gurobi/10.0.1-GCCcore-12.3.0
export GRB_LICENSE_FILE="../gurobi.lic"

i=${SLURM_ARRAY_TASK_ID}
$(head -$i problems/experiment_files/parameters_experiment_hewitt_keutchayan.txt | tail -1)
