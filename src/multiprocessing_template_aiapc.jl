using AlgorithmicCompetition
using Statistics
using DataFrames
using CSV
using Distributed
using Dates

version = "v0.0.2"
start_timestamp = now()
start_timestamp = Dates.format(start_timestamp, "yyyy-mm-dd HHMMSS")

if Sys.isapple()
    n_procs_ = 7 # up to 8 performance cores on m1 (7 workers + 1 main)

    n_parameter_iterations = 2
else
    n_procs_ = 60

    n_parameter_iterations = 40
end


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

@time exp_df = AlgorithmicCompetition.run_aiapc(;
    n_parameter_iterations = n_parameter_iterations,
    max_iter = Int(1e9),
    version = version,
    start_timestamp = start_timestamp,
)

rmprocs(_procs)

file_name = "simulation_results_aiapc_$(version)_$(start_timestamp).csv"
CSV.write(file_name, exp_df)
