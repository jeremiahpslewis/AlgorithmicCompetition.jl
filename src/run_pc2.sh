#!/bin/bash

# In order to run this script, do the following:
# TASK=DDDC bash src/run_pc2.sh
# or
# TASK=AIAPC bash src/run_pc2.sh


# NOTE: to self
# try using home directory for DEPOT and check runtime performance and storage limits: $HOME/.julia
# try using PackageCompiler.jl (see screenshots) as luster performance (/scratch parallel file system) is poor for lots of small files

module load lang       # loading the gateway module
module load JuliaHPC   # loading the latest JuliaHPC

cd /scratch/hpc-prf-irddcc || exit

[ ! -d 'AlgorithmicCompetition.jl' ] && git clone https://github.com/jeremiahpslewis/AlgorithmicCompetition.jl.git

cd AlgorithmicCompetition.jl || exit

export JULIA_DEPOT_PATH=/scratch/hpc-prf-irddcc/.julia

bash <(curl -Ls https://gist.githubusercontent.com/jeremiahpslewis/373e5e4d4d6faf1bf1a59ef9414019ca/raw/7e9166d3051687e2371b4ab98e6db1ac64432bef/sacct_tail.sh)

git pull

julia -e 'using Pkg; Pkg.activate("."); Pkg.update(); Pkg.instantiate(); Pkg.precompile()'

mkdir -p log # Log directory for slurm task output

if [[ "$TASK" == "DDDC" ]]; then
    sbatch src/dddc_slurm_sbatch.sh
elif [[ "$TASK" == "AIAPC" ]]; then
    sbatch src/aiapc_slurm_sbatch.sh
fi
