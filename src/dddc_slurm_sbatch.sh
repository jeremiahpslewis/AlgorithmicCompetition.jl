#!/bin/bash
#SBATCH --job-name=array_job
#SBATCH --output=log/%A_%a.out
#SBATCH --error=log/%A_%a.err
# SBATCH --array=1-10 # Number of iterations over all parameter sets
#SBATCH --mem-per-cpu=3200M
#SBATCH --cpus-per-task=64 # 128 core
#SBATCH --time=1:00:00 # For full run true value should be <13 hours
#SBATCH -p normal
#SBATCH --mail-user=irddcc1@mail.uni-paderborn.de # Where to send mail	

# For full version, N_GRID_INCREMENTS=100
export N_GRID_INCREMENTS=1
export N_PARAMETER_ITERATIONS=100 # Number of iterations over all parameter sets per job
export VERSION="2025-03-17-dddc-trembling-hand"
export JULIA_DEPOT_PATH=/scratch/hpc-prf-irddcc/.julia
export DEBUG=0
export LOG_DIR="/scratch/hpc-prf-irddcc/AlgorithmicCompetition.jl/log"
echo "Bash: Running DDDC with $N_GRID_INCREMENTS grid increments"

module reset
module load lang       # loading the gateway module
module load JuliaHPC   # loading the latest JuliaHPC

export PROJECTDIR="/scratch/hpc-prf-irddcc/AlgorithmicCompetition.jl"

julia --project=. src/dddc_slurm_batch.jl
