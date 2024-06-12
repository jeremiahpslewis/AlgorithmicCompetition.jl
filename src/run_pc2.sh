#!/bin/bash

# In order to run this script, do the following:
# TASK=DDDC bash src/run_pc2.sh 
# or
# TASK=AIAPC bash src/run_pc2.sh 

module load lang       # loading the gateway module
# module load JuliaHPC   # loading the latest JuliaHPC
module load Julia/1.10.4-linux-x86_64 # This should be fine, not using MPI for this project...

cd /scratch/hpc-prf-irddcc || exit
export JULIA_DEPOT_PATH=/scratch/hpc-prf-irddcc/v1/.julia

[ ! -d 'AlgorithmicCompetition.jl' ] && git clone https://github.com/jeremiahpslewis/AlgorithmicCompetition.jl.git

cd AlgorithmicCompetition.jl || exit

git pull

julia -e 'using Pkg; Pkg.activate("."); Pkg.update(); Pkg.instantiate()'

mkdir -p log # Log directory for slurm task output

if [[ "$TASK" == "DDDC" ]]; then
    sbatch src/dddc_slurm_sbatch.sh
elif [[ "$TASK" == "AIAPC" ]]; then
    sbatch src/aiapc_slurm_sbatch.sh
fi
