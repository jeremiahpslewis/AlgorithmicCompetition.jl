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
    n_parameter_iterations = 10,
    n_parameter_increments = 25,
    max_iter = Int(1e9), # TODO: increment to 1e9
    # max_alpha = 0.05,
    # max_beta = 0.5,
    # sample_fraction = 0.05,
)

rmprocs(_procs)

df = AlgorithmicCompetition.extract_sim_results(exp_list)
CSV.write("simulation_results_$start_timestamp.csv", df)

df = DataFrame(CSV.File("simulation_results_$start_timestamp.csv"))

using CairoMakie
using Chain
using DataFrameMacros
using AlgebraOfGraphics
using AlgorithmicCompetition:
    AIAPCHyperParameters, AIAPCEnv, CompetitionParameters, CompetitionSolution, profit_gain

competition_params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))
competition_solution = CompetitionSolution(competition_params)

env =
    AIAPCHyperParameters(
        Float64(0.1),
        Float64(1e-4),
        0.95,
        Int(1e7),
        competition_solution,
    ) |> AIAPCEnv

df_summary = @chain df begin
    @groupby(:α, :β)
    @combine(
        :Δ_π_bar = profit_gain(mean(:π_bar), env),
        :iterations_until_convergence = mean(:iterations_until_convergence)
    )
end

plt1 = @chain df_summary begin
    data(_) * mapping(:β, :α, :Δ_π_bar) * visual(Heatmap)
end

draw(plt1)

plt2 = @chain df_summary begin
    data(_) * mapping(:β, :α, :iterations_until_convergence) * visual(Heatmap)
end

draw(plt2)
