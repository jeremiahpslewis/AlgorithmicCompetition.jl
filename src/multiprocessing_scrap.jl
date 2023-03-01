using Distributed
using AlgorithmicCompetition:
    AlgorithmicCompetition,
    CompetitionParameters,
    CompetitionSolution,
    AIAPCHyperParameters,
    AIAPCEnv,
    Experiment,
    run_and_extract,
    run
using ReinforcementLearning
using Chain
using Statistics
using DataFrames
using CairoMakie
using DataFrameMacros
using CSV
using ProgressMeter
using BenchmarkTools

multiproc = true

using ParallelDataTransfer

# # TODO: Figure out why both players have identical average profits ALWAYS and add test?
@test mean([ex.avg_profit[1] == ex.avg_profit[2] for ex in exp_list]) < 0.1

α_result = [ex.α for ex in exp_list if !(ex isa Exception)]
β_result = [ex.β for ex in exp_list if !(ex isa Exception)]
iterations_until_convergence = [ex.iterations_until_convergence for ex in exp_list if !(ex isa Exception)]

avg_profit_result = [ex.avg_profit[1] for ex in exp_list if !(ex isa Exception)]

# α_result = [ex.α for ex in exp_list]
# β_result = [ex.β for ex in exp_list]
# avg_profit_result = [ex.avg_profit[1] for ex in exp_list]

df = DataFrame(α = α_result, β = β_result, π_bar = avg_profit_result, iterations_until_convergence = iterations_until_convergence)

CSV.write("simulation_results.csv", df)
plt_ = @chain df begin
    @groupby(:β, :α)
    @combine(:iterations_until_convergence = mean(:iterations_until_convergence))
    @combine(:heatmap = heatmap(:β, :α, :iterations_until_convergence)) 
end
save("test_convergence.png", plt_[1, :heatmap])

plt_ = @chain df @combine(:heatmap = heatmap(:β, :α, :π_bar))
save("test_profit.png", plt_[1, :heatmap])
