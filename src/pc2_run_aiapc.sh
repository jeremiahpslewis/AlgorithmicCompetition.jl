#!/bin/bash
tmux new -s aiapc

module load lang       # loading the gateway module
module load JuliaHPC   # loading the latest JuliaHPC

cd /scratch/hpc-prf-irddcc || exit
export JULIA_DEPOT_PATH=/scratch/hpc-prf-irddcc/.julia

[ ! -d 'AlgorithmicCompetition.jl' ] && git clone https://github.com/jeremiahpslewis/AlgorithmicCompetition.jl.git

cd AlgorithmicCompetition.jl || exit

julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'

mkdir -p log # Log directory for slurm task output
sbatch src/hpc_slurm.sh


## Viz Analysis Script
julia --project=.
