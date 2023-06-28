using CairoMakie
using Chain
using DataFrameMacros
using AlgebraOfGraphics
using AlgorithmicCompetition:
    AIAPCHyperParameters, AIAPCEnv, CompetitionParameters, CompetitionSolution, profit_gain
using CSV
using DataFrames
using Statistics

file_name = "simulation_results_2023-06-21T00:01:53.187.csv"
df = DataFrame(CSV.File(file_name))

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
    @transform(:β = round(:β, digits = 7), :α = round(:α, digits = 2))
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
