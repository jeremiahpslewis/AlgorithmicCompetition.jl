using AlgorithmicCompetition
using Statistics
using DataFrames
using CSV
using Distributed
using Dates

version = 0.2
start_timestamp = now()

if Sys.isapple()
    n_procs_ = 7 # up to 8 performance cores on m1 (7 workers + 1 main)

    n_parameter_iterations = 15
    n_grid_increments = 25
else
    n_procs_ = 63

    n_parameter_iterations = 5
    n_grid_increments = 25
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

@time exp_list = AlgorithmicCompetition.run_dddc(;
    n_parameter_iterations = n_parameter_iterations,
    max_iter = Int(1e9),
    n_grid_increments = n_grid_increments,
)

rmprocs(_procs)

file_name = "simulation_results_v$(version)_dddc_$(start_timestamp).csv"
exp_list_ = AlgorithmicCompetition.DDDCSummary[exp_list...]
df = AlgorithmicCompetition.extract_sim_results(exp_list_)
CSV.write(file_name, df)
