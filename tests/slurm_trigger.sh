#!/bin/bash
#SBATCH --job-name=SIMBAtest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=0-00:01:00
#SBATCH --array=0-2

# Load the necessary modules
module load anaconda3/5.2.0
eval "$(conda shell.bash hook)"
conda activate lag_conda

# Run your Python script with $SLURM_ARRAY_TASK_ID as argument
python simba_runtest.py -i $SLURM_ARRAY_TASK_ID
