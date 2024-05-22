tmux new -s aiapc

module load lang       # loading the gateway module
module load JuliaHPC   # loading the latest JuliaHPC

cd /scratch/hpc-prf-irddcc
export JULIA_DEPOT_PATH=/scratch/hpc-prf-irddcc/.julia

git clone https://github.com/jeremiahpslewis/AlgorithmicCompetition.jl.git

cd AlgorithmicCompetition.jl

sbatch src/hpc_slurm.sh
