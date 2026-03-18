#!/bin/bash
# Set Job Requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=rome
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=j.slootweg@vu.nl
#SBATCH --time=20:05:00

module load 2022
module load Anaconda3/2022.05
module load Gurobi/9.5.2-GCCcore-11.3.0

export GRB_LICENSE_FILE="${HOME}/gurobi.lic"

mkdir "$TMPDIR"/models
mkdir "$TMPDIR"/results
mkdir "$TMPDIR"/tourists

source activate tourists

for INSTANCE_NUMBER in 0
do
  for EVENTS in 5
  do
	    for NUMBER_OF_VISITORS in 50
	    do
	      for VISITOR_INSTANCE in 0 1 2 3
	      do
	        python $HOME/tourists/main.py "${HOME}/tourists/instances/${EVENTS}_${NUMBER_OF_VISITORS}_${INSTANCE_NUMBER}" $VISITOR_INSTANCE "${TMPDIR}/models/${EVENTS}_${NUMBER_OF_VISITORS}_${INSTANCE_NUMBER}_${VISITOR_INSTANCE}.lp" "${TMPDIR}/results/${EVENTS}_${NUMBER_OF_VISITORS}_${INSTANCE_NUMBER}_${VISITOR_INSTANCE}.csv" "${TMPDIR}/tourists/${EVENTS}_${NUMBER_OF_VISITORS}_${INSTANCE_NUMBER}_${VISITOR_INSTANCE}.csv"
	      done
	    done
	done
done

cp -r "$TMPDIR"/models $HOME/tourists
cp -r "$TMPDIR"/results $HOME/tourists
cp -r "$TMPDIR"/tourists $HOME/tourists