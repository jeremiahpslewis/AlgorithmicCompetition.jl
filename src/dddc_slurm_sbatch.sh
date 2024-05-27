#!/bin/bash
#SBATCH --job-name=array_job
#SBATCH --output=log/array_job_%A_%a.out
#SBATCH --error=log/array_job_%A_%a.err
#SBATCH --array=1-1000 # 1-1000 # Number of iterations over all parameter sets
#SBATCH --mem-per-cpu=1500M
#SBATCH --cpus-per-task=1
#SBATCH --time=2-12 # For full run true value should be <45 hours, e.g. 1-12
#SBATCH -p normal
#SBATCH --mail-user=irddcc1@mail.uni-paderborn.de   # Where to send mail	

# Number of iterations per parameter set; for parallelized full run, 1 iteration over 10k parameters
export N_ITERATIONS=1 
export DEBUG=0
export VERSION="2024-05-22-debug"
julia --project=. src/dddc_slurm_batch.jl
