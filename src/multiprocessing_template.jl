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
    n_parameter_iterations = 25,
    n_parameter_increments = 50,
    max_iter = Int(1e9), # TODO: increment to 1e9
)

rmprocs(_procs)

df = AlgorithmicCompetition.extract_sim_results(exp_list)
CSV.write("simulation_results_$start_timestamp.csv", df)

df = DataFrame(CSV.File("simulation_results_$start_timestamp.csv"))

using CairoMakie
using Chain
using DataFrameMacros
using AlgebraOfGraphics

df_summary = @chain df begin
    @groupby(:α, :β)
    @combine(:Δ_π_bar = mean(:π_bar),
               :iterations_until_convergence = mean(:iterations_until_convergence))
end

plt1 = @chain df_summary begin
    data(_) *
    mapping(:β, :α, :Δ_π_bar) *
    visual(Heatmap)
end

draw(plt1)

plt2 = @chain df_summary begin
    data(_) *
    mapping(:β, :α, :iterations_until_convergence => log10) *
    visual(Heatmap)
end

draw(plt2)

