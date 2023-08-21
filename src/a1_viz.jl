using CairoMakie
using Chain
using DataFrameMacros
using AlgebraOfGraphics
using AlgorithmicCompetition:
    AIAPCHyperParameters, AIAPCEnv, CompetitionParameters, CompetitionSolution, profit_gain
using CSV
using DataFrames
using Statistics

folder_name = "aiapc_v0.0.2_data"
df = DataFrame(CSV.File(file_name))

df_ = DataFrame.(CSV.File.(readdir(folder_name, join = true)))
df_ = vcat(df_...)

competition_params_dict = Dict(
    :high => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
    :low => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
)
competition_solution_dict =
    Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

hyperparams = AIAPCHyperParameters(
    Float64(0.1),
    Float64(1e-4),
    0.95,
    Int(1e7),
    competition_solution_dict,
)
env = AIAPCEnv(hyperparams)

@chain df @subset(:α == 0.0025)

df_summary = @chain df begin
    @groupby(:α, :β)
    @combine(
        :Δ_π_bar = profit_gain(mean(:π_bar), env),
        :iterations_until_convergence = mean(:iterations_until_convergence)
    )
end

plt0 = draw_price_diagnostic(hyperparams)
draw(
    plt0,
    axis = (
        title = "Profit Levels across Price Options",
        subtitle = "(Solid line is profit for symmetric prices, shaded region shows range based on price options)",
        xlabel = "Competitor's Price Choice",
    ),
)


plt1 = @chain df_summary begin
    data(_) * mapping(:β, :α, :Δ_π_bar) * visual(Heatmap)
end

draw(plt1)

plt2 = @chain df_summary begin
    data(_) * mapping(:β, :α, :iterations_until_convergence) * visual(Heatmap)
end

draw(plt2)
