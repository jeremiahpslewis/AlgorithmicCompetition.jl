#!/bin/bash
#SBATCH --job-name=array_job
#SBATCH --output=log/%A_%a.out
#SBATCH --error=log/%A_%a.err
#SBATCH --array=1-4 # 1-1000 # Number of iterations over all parameter sets
#SBATCH --mem-per-cpu=1000M
#SBATCH --cpus-per-task=50
#SBATCH --time=3:30:00 # For full run true value should be <45 hours, e.g. 1-12
#SBATCH -p normal
#SBATCH --mail-user=irddcc1@mail.uni-paderborn.de   # Where to send mail	

# Number of iterations per parameter set; for parallelized full run, 1 iteration over 10k parameters
export N_ITERATIONS=1
export DEBUG=1
export VERSION="2024-05-30-dddc-trial"
export N_CORES=50
export JULIA_DEPOT_PATH=/scratch/hpc-prf-irddcc/v1/.julia
julia --project=. src/dddc_slurm_batch.jl
