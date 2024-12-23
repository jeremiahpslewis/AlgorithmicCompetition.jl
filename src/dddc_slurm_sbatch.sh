#!/bin/bash
#SBATCH --job-name=array_job
#SBATCH --output=log/%A_%a.out
#SBATCH --error=log/%A_%a.err
#SBATCH --array=1-2 # Number of iterations over all parameter sets
#SBATCH --mem-per-cpu=1600M
#SBATCH --cpus-per-task=128 # 128 core
#SBATCH --time=20:00:00 # For full run true value should be <13 hours
#SBATCH -p normal
#SBATCH --mail-user=irddcc1@mail.uni-paderborn.de   # Where to send mail	

# For full version, N_GRID_INCREMENTS=100
export N_GRID_INCREMENTS=10
export N_PARAMETER_ITERATIONS=1 # Number of iterations over all parameter sets per job
export VERSION="2024-12-23-dddc-full-test"
export JULIA_DEPOT_PATH=/scratch/hpc-prf-irddcc/.julia
export DEBUG=0
echo "Bash: Running DDDC with $N_GRID_INCREMENTS grid increments"

module reset
module load lang       # loading the gateway module
module load JuliaHPC   # loading the latest JuliaHPC

export SINGULARITYENV_SLURM_ARRAY_TASK_ID="$SLURM_ARRAY_TASK_ID"
export SINGULARITYENV_SLURM_ARRAY_JOB_ID="$SLURM_ARRAY_JOB_ID"
export SINGULARITYENV_SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

# singularity run algorithmiccompetition.jl_main.sif julia --project=/algcomp /algcomp/src/dddc_slurm_batch.jl

julia --project=. src/dddc_slurm_batch.jl
