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
    n_grid_increments = 100,
)

    frequency_high_demand_range = Float64.(range(0, 1, n_grid_increments))
    competition_params_dict = Dict(
        :high => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
        :low => CompetitionParameters(-0.25, 0, (2, 2), (1, 1)), # Akin to Calvano 2020 Stochastic Demand
    )
    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

    δ = 0.95

    data_demand_digital_param_set = [DataDemandDigitalParams(
        low_signal_quality_level = 0.75,
        high_signal_quality_level = 1.0,
        signal_quality_is_high = [true, false],
        frequency_high_demand = frequency_high_demand,
    ) for frequency_high_demand in frequency_high_demand_range]

    hyperparameter_vect = [DDDCHyperParameters(
        α,
        β,
        δ,
        max_iter,
        competition_solution_dict,
        ;
        convergence_threshold = Int(1e5),
    ) for data_demand_digital_params in data_demand_digital_param_set]

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

