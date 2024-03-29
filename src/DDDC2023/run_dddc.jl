import ProgressMeter: @showprogress
using Distributed
using Random
using StatsBase

"""
    run_dddc(
        n_parameter_iterations = 1,
        max_iter = Int(1e9),
        convergence_threshold = Int(1e5),
        n_grid_increments = 100,
    )

Run DDDC, given a configuration for a set of experiments.
"""
function run_dddc(;
    n_parameter_iterations = 1,
    max_iter = Int(1e9),
    convergence_threshold = Int(1e5),
    n_grid_increments = 100,
)
    frequency_high_demand_range = Float64.(range(0.5, 1, n_grid_increments + 1))
    weak_signal_quality_level_range = Float64.(range(0.5, 1.0, 6))
    # Capture 'extreme' values
    weak_signal_quality_level_range = [weak_signal_quality_level_range...]
    signal_quality_vect = [[true, false], [false, false]]
    competition_params_dict = Dict(
        :low => CompetitionParameters(0.25, 0.25, (2, 2), (1, 1)),
        :high => CompetitionParameters(0.25, -0.25, (2, 2), (1, 1)), # Parameter values aligned with Calvano 2020 Stochastic Demand case
    )

    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

    α = Float64(0.15)
    β = Float64(4e-1)
    δ = 0.95

    data_demand_digital_param_set = [
        DataDemandDigitalParams(
            weak_signal_quality_level = weak_signal_quality_level,
            strong_signal_quality_level = 1.0,
            signal_is_strong = shuffle(signal_quality_players),
            frequency_high_demand = frequency_high_demand,
        ) for frequency_high_demand in frequency_high_demand_range for
        signal_quality_players in signal_quality_vect for
        weak_signal_quality_level in weak_signal_quality_level_range
    ]

    hyperparameter_vect = [
        DDDCHyperParameters(
            α,
            β,
            δ,
            max_iter,
            competition_solution_dict,
            data_demand_digital_params;
            convergence_threshold = convergence_threshold,
        ) for data_demand_digital_params in data_demand_digital_param_set
    ]

    # Shuffle hyperparameter_vect, extend according to number of repetitions
    hyperparameter_vect = shuffle(repeat(hyperparameter_vect, n_parameter_iterations))
    exp_list_ = DDDCSummary[]
    println(
        "About to run $(length(hyperparameter_vect) ÷ n_parameter_iterations) parameter settings, each $n_parameter_iterations times",
    )
    exp_list = @showprogress pmap(run_and_extract, hyperparameter_vect; on_error = identity)
    append!(exp_list_, exp_list)

    return exp_list_
end
