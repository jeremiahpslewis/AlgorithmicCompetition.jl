#!/bin/bash
#SBATCH --job-name=array_job
#SBATCH --output=log/%A_%a.out
#SBATCH --error=log/%A_%a.err
#SBATCH --array=1 # Number of iterations over all parameter sets
#SBATCH --mem-per-cpu=3gb
#SBATCH --cpus-per-task=64 # 128 core
#SBATCH --time=1:00:00 # For full run true value should be <13 hours
#SBATCH --partition=standard

module load julia

# For full version, N_GRID_INCREMENTS=100
export N_GRID_INCREMENTS=0 # Number of grid increments
export N_PARAMETER_ITERATIONS=1 # Number of iterations over all parameter sets per job
export VERSION="2025-03-19-dddc-trembling-hand"
export DEBUG=0
export JULIA_DEPOT_PATH=/lustre/wiwi/lewisjps/.julia
export LOG_DIR="/lustre/wiwi/lewisjps/AlgorithmicCompetition.jl/log"

echo "Bash: Running DDDC with $N_GRID_INCREMENTS grid increments"

cd /lustre/wiwi/lewisjps/AlgorithmicCompetition.jl || exit

julia --project=. src/dddc_slurm_batch.jl
