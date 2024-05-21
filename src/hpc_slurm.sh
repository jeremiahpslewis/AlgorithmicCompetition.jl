#!/bin/bash
#SBATCH --job-name=array_job
#SBATCH --output=array_job_%A_%a.out
#SBATCH --error=array_job_%A_%a.err
#SBATCH --array=1-1000 # after % is the number of simultaneous jobs
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00:00 # For full run this is 4.5 hours
#SBATCH -p normal
#SBATCH --mail-user=irddcc1@mail.uni-paderborn.de   # Where to send mail	

export N_ITERATIONS=1000 # Number of iterations per parameter set; for full run, this is 1000
export VERSION="2024-05-21-final"
julia src/slurm_batch.jl
