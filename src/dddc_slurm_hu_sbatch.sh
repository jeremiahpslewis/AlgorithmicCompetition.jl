#!/bin/bash
#SBATCH --job-name=array_job
#SBATCH --output=log/%A_%a.out
#SBATCH --error=log/%A_%a.err
#SBATCH --array=1-300 # Number of iterations over all parameter sets
#SBATCH --mem-per-cpu=1600M
#SBATCH --cpus-per-task=16 # 128 core
#SBATCH --time=3:55:00 # For full run true value should be <13 hours
#SBATCH --partition standard

# For full version, N_GRID_INCREMENTS=100
export N_GRID_INCREMENTS=20 # Number of grid increments
export N_PARAMETER_ITERATIONS=2 # Number of iterations over all parameter sets per job
export VERSION="2024-12-05-dddc-hu-test"
export DEBUG=0
echo "Bash: Running DDDC with $N_GRID_INCREMENTS grid increments"

export APPTAINERENV_SLURM_ARRAY_TASK_ID="$SLURM_ARRAY_TASK_ID"
export APPTAINERENV_SLURM_ARRAY_JOB_ID="$SLURM_ARRAY_JOB_ID"
export APPTAINERENV_SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

apptainer run algorithmiccompetition.jl_main.sif julia --project=/algcomp /algcomp/src/dddc_slurm_batch.jl

