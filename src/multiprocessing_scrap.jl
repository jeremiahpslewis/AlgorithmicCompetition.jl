using Distributed
using AlgorithmicCompetition:
    AlgorithmicCompetition,
    CompetitionParameters,
    CompetitionSolution,
    CalvanoHyperParameters,
    CalvanoEnv,
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
using JET
using BenchmarkTools

multiproc = false

using ParallelDataTransfer

if multiproc
    _procs = addprocs(
        7,
        topology = :master_worker,
        exeflags = ["--threads=1", "--project=$(Base.active_project())"],
    )

    @everywhere begin
        using Pkg
        Pkg.instantiate()
        using AlgorithmicCompetition:
            AlgorithmicCompetition, CalvanoHyperParameters, CalvanoEnv, run
    end
end

competition_params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])

competition_solution = CompetitionSolution(competition_params)

n_increments = 100
max_iter = Int(1e6) # Should be 1e9
α_ = range(0.025, 0.25, n_increments)
β_ = range(1.25e-8, 2e-5, n_increments)
δ = 0.95

hyperparameter_vect = [
    CalvanoHyperParameters(α, β, δ, max_iter, competition_solution) for α in α_ for β in β_
]

@btime run_and_extract(hyperparameter_vect[1]; stop_on_convergence=false)

a = run(hyperparameter_vect[1]; stop_on_convergence=false)




# if multiproc
#     exp_list = @showprogress pmap(run_and_extract, hyperparameter_vect; on_error=identity)
# end

# # TODO: Figure out why both players have identical average profits ALWAYS and add test?
# mean([ex.avg_profit[1] == ex.avg_profit[2] for ex in exp_list])

# α_result = [ex.α for ex in exp_list if !(ex isa Exception)]
# β_result = [ex.β for ex in exp_list if !(ex isa Exception)]
# avg_profit_result = [ex.avg_profit[1] for ex in exp_list if !(ex isa Exception)]



# # TODO: Figure out why both players have identical average profits ALWAYS and add test?
# mean([ex.avg_profit[1] == ex.avg_profit[2] for ex in exp_list])

# α_result = [ex.α for ex in exp_list]
# β_result = [ex.β for ex in exp_list]
# avg_profit_result = [ex.avg_profit[1] for ex in exp_list]


# df = DataFrame(α = α_result, β = β_result, π_bar = avg_profit_result)

# CSV.write("simulation_results.csv", df)
# plt_ = @chain df @combine(:heatmap = heatmap(:β, :α, :π_bar))

# save("test.png", plt_[1, :heatmap])
