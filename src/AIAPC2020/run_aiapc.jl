import ProgressMeter: @showprogress
using Distributed
using Random
using StatsBase

const α_range = Float64.(range(0.0025, 0.25, 100))
const β_range = Float64.(range(0.005, 0.5, 100))

function run_aiapc(;
    n_parameter_iterations = 1,
    max_iter = Int(1e9),
    convergence_threshold = Int(1e5),
    max_alpha = 0.25,
    max_beta = 2,
    sample_fraction = 1,
)
    competition_params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))
    competition_solution = CompetitionSolution(competition_params)

    # Filter params based on max_alpha and max_beta values
    α_ = α_range[α_range .<=max_alpha]
    β_ = α_range[β_range .<=max_beta]

    β_ = β_ * 1e-5 # rescale beta
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

    # Sample fraction of α & β coordinates
    hyperparameter_vect = sample(
        hyperparameter_vect,
        Int(floor(length(hyperparameter_vect) * sample_fraction)),
    )

    # Shuffle hyperparameter_vect, extend according to number of repetitions
    hyperparameter_vect = shuffle(repeat(hyperparameter_vect, n_parameter_iterations))
    exp_list_ = AIAPCSummary[]
    println(
        "About to run $(length(hyperparameter_vect) / n_parameter_iterations) parameter settings, each $n_parameter_iterations times",
    )
    exp_list = @showprogress pmap(run_and_extract, hyperparameter_vect; on_error = identity)
    append!(exp_list_, exp_list)

    return exp_list_
end
