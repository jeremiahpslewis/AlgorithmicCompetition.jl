#!/bin/bash
#SBATCH --job-name=array_job
#SBATCH --output=log/%A_%a.out
#SBATCH --error=log/%A_%a.err
#SBATCH --array=1-2 # Number of iterations over all parameter sets
#SBATCH --mem-per-cpu=1600M
#SBATCH --cpus-per-task=16 # 128 core
#SBATCH --time=24:00:00 # For full run true value should be <13 hours
#SBATCH --partition standard

# For full version, N_GRID_INCREMENTS=100
export N_GRID_INCREMENTS=20 # Number of grid increments
export N_PARAMETER_ITERATIONS=1 # Number of iterations over all parameter sets per job
export VERSION="2025-03-10-7price-config"
export DEBUG=1
echo "Bash: Running DDDC with $N_GRID_INCREMENTS grid increments"

module load julia

julia --project=. ~/AlgorithmicCompetition.jl/src/dddc_slurm_batch.jl
