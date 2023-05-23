import ProgressMeter: @showprogress
using Distributed
using Random

function run_aiapc(;
    n_parameter_iterations = 1,
    n_parameter_increments = 100,
    max_iter = Int(1e9),
    convergence_threshold = Int(1e5),
)
    competition_params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))
    competition_solution = CompetitionSolution(competition_params)

    α_ = Float64.(range(0.025, 0.25, n_parameter_increments))
    β_ = Float64.(range(1.25e-8, 2e-5, n_parameter_increments))
    δ = 0.95

    hyperparameter_vect = [
        AIAPCHyperParameters(
            α,
            β,
            δ,
            max_iter,
            competition_solution;
            convergence_threshold = convergence_threshold,
        ) for α in α_ for β in β_
    ]
    exp_list_ = AIAPCSummary[]
    println("About to run $(length(hyperparameter_vect)) parameter settings, each $n_parameter_iterations times")
    exp_list =
        @showprogress pmap(run_and_extract, shuffle(repeat(hyperparameter_vect, n_parameter_iterations)); on_error = identity)
    append!(exp_list_, exp_list)

    return exp_list_
end
