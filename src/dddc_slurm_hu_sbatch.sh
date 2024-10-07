#!/bin/bash
#SBATCH --job-name=array_job
#SBATCH --output=log/%A_%a.out
#SBATCH --error=log/%A_%a.err
#SBATCH --array=1-5 # Number of iterations over all parameter sets
#SBATCH --mem-per-cpu=1600M
#SBATCH --cpus-per-task=64 # 128 core
#SBATCH --time=1:00:00 # For full run true value should be <13 hours
#SBATCH -p std

# For full version, N_GRID_INCREMENTS=100
export N_GRID_INCREMENTS=20
export N_PARAMETER_ITERATIONS=1 # Number of iterations over all parameter sets per job
export VERSION="2024-10-07-dddc-hu-test"
export DEBUG=1
echo "Bash: Running DDDC with $N_GRID_INCREMENTS grid increments"

export APPTAINERENV_SLURM_ARRAY_TASK_ID="$SLURM_ARRAY_TASK_ID"
export APPTAINERENV_SLURM_ARRAY_JOB_ID="$SLURM_ARRAY_JOB_ID"
export APPTAINERENV_SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

apptainer run algorithmiccompetition.jl_main.sif julia --project=/algcomp /algcomp/src/dddc_slurm_batch.jl

