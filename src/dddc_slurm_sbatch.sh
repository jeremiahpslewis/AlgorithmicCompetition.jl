#!/bin/bash
#SBATCH --job-name=array_job
#SBATCH --output=log/%A_%a.out
#SBATCH --error=log/%A_%a.err
#SBATCH --array=1-2 # Number of iterations over all parameter sets
#SBATCH --mem-per-cpu=1600M
#SBATCH --cpus-per-task=1 # 128 core
#SBATCH --time=20:00:00 # For full run true value should be <13 hours
#SBATCH -p normal
#SBATCH --mail-user=irddcc1@mail.uni-paderborn.de   # Where to send mail	

# For full version, N_GRID_INCREMENTS=100
export N_GRID_INCREMENTS=20
export VERSION="2024-07-23-dddc-full-strong-weak-grid"
export JULIA_DEPOT_PATH=/scratch/hpc-prf-irddcc/.julia
export DEBUG=0
echo "Bash: Running DDDC with $N_GRID_INCREMENTS grid increments"
# julia +1.10 --project=. src/dddc_slurm_batch.jl
# julia +1.10 -e 'using Pkg; Pkg.activate("."); Pkg.update(); Pkg.instantiate()'
julia +1.10 -e 'println("Hello, World!")'

