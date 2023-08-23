import ProgressMeter: @showprogress
using Distributed
using Random
using StatsBase
using DataFrames
using CSV
using Dates

function build_hyperparameter_set(
    α_vect,
    β_vect,
    δ,
    max_iter,
    competition_solution_dict,
    convergence_threshold,
    n_parameter_iterations,
)
    hyperparameter_vect = [
        AIAPCHyperParameters(
            α,
            β,
            δ,
            max_iter,
            competition_solution_dict;
            convergence_threshold = convergence_threshold,
        ) for α in α_vect for β in β_vect
    ]

    # Shuffle hyperparameter_vect, extend according to number of repetitions
    hyperparameter_vect = shuffle(repeat(hyperparameter_vect, n_parameter_iterations))
    return hyperparameter_vect
end


"""
    run_aiapc(
        n_parameter_iterations = 1,
        max_iter = Int(1e9),
        convergence_threshold = Int(1e5),
        max_alpha = 0.25,
        max_beta = 2,
        sample_fraction = 1,
    )

Run AIAPC, given a configuration for a set of experiments.
"""
function run_aiapc(;
    n_parameter_iterations = 1,
    max_iter = Int(1e9),
    convergence_threshold = Int(1e5),
    α_range = Float64.(range(0.0025, 0.25, 100)),
    β_range = Float64.(range(0.02, 2, 100)),
    version = "v0.0.0",
    start_timestamp = now(),
)
    competition_params_dict = Dict(
        :high => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
    )
    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

    δ = 0.95

    hyperparameter_vect = build_hyperparameter_set(
        α_range,
        β_range,
        δ,
        max_iter,
        competition_solution_dict,
        convergence_threshold,
        1,
    )

    println(
        "About to run $(length(hyperparameter_vect)) parameter settings, each $n_parameter_iterations times",
    )

    folder_name = "aiapc_$(version)_$(start_timestamp)"
    mkpath(folder_name)

    for i = 1:n_parameter_iterations
        println("Parameter iteration $i of $n_parameter_iterations")
        file_name = "$(folder_name)/simulation_results_aiapc_$(i).csv"


        exp_list_ = AIAPCSummary[]
        exp_list =
            @showprogress pmap(run_and_extract, hyperparameter_vect; on_error = identity)

        df = AlgorithmicCompetition.extract_sim_results(exp_list)
        CSV.write(file_name, df)
    end

    exp_df = DataFrame.(CSV.File.(readdir(folder_name, join = true)))
    exp_df = vcat(exp_df...)

    return exp_df
end
