#!/bin/bash

# In order to run this script, do the following:
# TASK=DDDC bash src/run_pc2.sh
# or
# TASK=AIAPC bash src/run_pc2.sh

export APPTAINER_CACHEDIR=/lustre/department/lewisjps/apptainer

cd /lustre/wiwi/lewisjps || exit

[ ! -d 'AlgorithmicCompetition.jl' ] && git clone https://github.com/jeremiahpslewis/AlgorithmicCompetition.jl.git

cd AlgorithmicCompetition.jl || exit

git pull

if [ -f algorithmiccompetition.jl_main.sif ]; then
    rm algorithmiccompetition.jl_main.sif
fi

apptainer pull docker://ghcr.io/jeremiahpslewis/algorithmiccompetition.jl:main

mkdir -p log # Log directory for slurm task output

if [[ "$TASK" == "DDDC" ]]; then
    sbatch src/dddc_slurm_sbatch.sh
elif [[ "$TASK" == "AIAPC" ]]; then
    sbatch src/aiapc_slurm_sbatch.sh
fi
