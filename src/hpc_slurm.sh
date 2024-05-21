#!/bin/bash
#SBATCH --job-name=array_job
#SBATCH --output=array_job_%A_%a.out
#SBATCH --error=array_job_%A_%a.err
#SBATCH --array=1-10    # 000%1000, after % is the number of simultaneous jobs
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH -p normal
#SBATCH --mail-user=irddcc1@mail.uni-paderborn.de   # Where to send mail	

julia src/slurm_batch.jl
