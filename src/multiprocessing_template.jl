using AlgorithmicCompetition
using Statistics
using DataFrames
using CSV
using Distributed
using Dates

start_timestamp = now()
n_procs_ = 7 # up to 8 performance cores on m1 (7 workers + 1 main)

_procs = addprocs(
    n_procs_,
    topology = :master_worker,
    exeflags = ["--threads=1", "--project=$(Base.active_project())"],
)

@everywhere begin
    using Pkg
    Pkg.instantiate()
    using AlgorithmicCompetition: run_and_extract
end

@time exp_list = AlgorithmicCompetition.run_aiapc(;
    n_parameter_iterations = 1,
    max_iter = Int(1e9), # TODO: increment to 1e9
    # max_alpha = 0.05,
    # max_beta = 0.5,
    sample_fraction = 1,
)

rmprocs(_procs)

file_name = "simulation_results_$start_timestamp.csv"
df = AlgorithmicCompetition.extract_sim_results(exp_list)
CSV.write(file_name, df)
