#!/bin/bash

# In order to run this script, do the following:
# TASK=DDDC bash src/run_hu_hpc.sh
# or
# TASK=AIAPC bash src/run_hu_hpc.sh

module load julia

export JULIA_DEPOT_PATH=/lustre/wiwi/lewisjps/.julia

cd /lustre/wiwi/lewisjps || exit

[ ! -d 'AlgorithmicCompetition.jl' ] && git clone https://github.com/jeremiahpslewis/AlgorithmicCompetition.jl.git 

cd AlgorithmicCompetition.jl || exit

git pull

julia -e 'using Pkg; Pkg.activate("."); Pkg.update(); Pkg.instantiate(); Pkg.precompile()'

mkdir -p log # Log directory for slurm task output

if [[ "$TASK" == "DDDC" ]]; then
    sbatch src/dddc_slurm_hu_sbatch.sh
elif [[ "$TASK" == "AIAPC" ]]; then
    sbatch src/aiapc_slurm_sbatch.sh
fi
