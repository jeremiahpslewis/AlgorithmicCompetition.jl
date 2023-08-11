import ProgressMeter: @showprogress
using Distributed
using Random
using StatsBase

"""
    run_dddc(
        n_parameter_iterations = 1,
        max_iter = Int(1e9),
        convergence_threshold = Int(1e5),
    )

Run AIAPC, given a configuration for a set of experiments.
"""
function run_dddc(;
    n_parameter_iterations = 1,
    max_iter = Int(1e9),
    convergence_threshold = Int(1e5),
    # max_alpha = 0.25,
    # max_beta = 2,
)
    competition_params_dict = Dict(
        :high => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
    )
    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

    δ = 0.95

    # NOTE: low quality probability 0.5+x, high quality boost, is high quality signal, high demand freq
    data_demand_digital_params = DataDemandDigitalParams(
        low_signal_quality_level = 0.99,
        high_signal_quality_boost = 0.005,
        signal_quality_is_high = [true, false],
        frequency_high_demand = 0.9,
    )

    hyperparams = DDDCHyperParameters(
        α,
        β,
        δ,
        max_iter,
        competition_solution_dict,
        data_demand_digital_params;
        convergence_threshold = Int(1e5),
    )
    # Sample fraction of α & β coordinates
    hyperparameter_vect = sample(
        hyperparameter_vect,
        Int(floor(length(hyperparameter_vect) * sample_fraction)),
    )

    # Shuffle hyperparameter_vect, extend according to number of repetitions
    hyperparameter_vect = shuffle(repeat(hyperparameter_vect, n_parameter_iterations))
    exp_list_ = AIAPCSummary[]
    println(
        "About to run $(length(hyperparameter_vect) ÷ n_parameter_iterations) parameter settings, each $n_parameter_iterations times",
    )
    exp_list = @showprogress pmap(run_and_extract, hyperparameter_vect; on_error = identity)
    append!(exp_list_, exp_list)

    return exp_list_
end

