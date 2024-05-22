#!/bin/bash
#SBATCH --job-name=array_job
#SBATCH --output=array_job_%A_%a.out
#SBATCH --error=array_job_%A_%a.err
#SBATCH --array=233,248,258,259,265,273,284,288,295,296,302,333,339,900 # 1-1000 # after % is the number of simultaneous jobs
#SBATCH --mem-per-cpu=1500M
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00 # For full run true value should be <13 hours
#SBATCH -p normal
#SBATCH --mail-user=irddcc1@mail.uni-paderborn.de   # Where to send mail	

# Number of iterations per parameter set; for parallelized full run, 1 iteration over 10k parameters
export N_ITERATIONS=1 

export VERSION="2024-05-22"
julia src/slurm_batch.jl
