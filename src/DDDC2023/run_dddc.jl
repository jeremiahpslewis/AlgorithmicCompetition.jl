import ProgressMeter: @showprogress
using Distributed
using Random
using StatsBase
using Arrow

"""
    run_dddc(
        n_parameter_iterations = 1,
        max_iter = Int(1e9),
        convergence_threshold = Int(1e5),
        n_grid_increments = 100,
        version = "v0.0.0",
        start_timestamp = now(),
        batch_size = 1,
        batch_metadata = (SLURM_ARRAY_JOB_ID = 0, SLURM_ARRAY_TASK_ID = 0),
        debug = false,
    )

Run DDDC, given a configuration for a set of experiments.
"""
function run_dddc(;
    n_parameter_iterations = 1,
    max_iter = Int(1e9),
    convergence_threshold = Int(1e5),
    n_grid_increments = 100,
    version = "v0.0.0",
    start_timestamp = now(),
    batch_size = 1,
    batch_metadata = (SLURM_ARRAY_JOB_ID = 0, SLURM_ARRAY_TASK_ID = 0),
    debug = false,
)
    # signal_quality_vect = [[true, false], [false, false]]
    signal_quality_vect = [[true, false]] # With signal_quality_range over both weak and strong, [false, false] case is redundant

    frequency_high_demand_range = Float64.(range(0.0, 1.0, n_grid_increments + 1))
    signal_quality_level_range = Float64.(range(0.5, 1.0, n_grid_increments + 1))

    if debug
        frequency_high_demand_range = frequency_high_demand_range[1:10:end]
        signal_quality_level_range = signal_quality_level_range[1:10:end]
    end

    competition_params_dict = Dict(
        :low => CompetitionParameters(0.25, 0.25, (2, 2), (1, 1)),
        :high => CompetitionParameters(0.25, -0.25, (2, 2), (1, 1)), # Parameter values aligned with Calvano 2020 Stochastic Demand case
    )

    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

    α = Float64(0.15)
    ν_ = 20.0 # From Calvano 2020
    frequency_high_demand = 0.3
    weak_signal_quality_level = 0.1
    δ = 0.95

    data_demand_digital_param_set = [
        DataDemandDigitalParams(
            weak_signal_quality_level = weak_signal_quality_level,
            strong_signal_quality_level = strong_signal_quality_level,
            signal_is_strong = signal_quality_players,
            frequency_high_demand = frequency_high_demand,
        ) for frequency_high_demand in frequency_high_demand_range for
        signal_quality_players in signal_quality_vect for
        weak_signal_quality_level in signal_quality_level_range for
        strong_signal_quality_level in signal_quality_level_range if
        weak_signal_quality_level <= strong_signal_quality_level
    ]

    hyperparameter_vect = [
        DDDCHyperParameters(
            α,
            ν_inverse(
                15,
                2,
                1,
                ν_tilde(ν_, frequency_high_demand, weak_signal_quality_level),
            ), # \beta value
            δ,
            max_iter,
            competition_solution_dict,
            data_demand_digital_params;
            convergence_threshold = convergence_threshold,
        ) for data_demand_digital_params in data_demand_digital_param_set
    ]

    # Shuffle hyperparameter_vect, extend according to number of repetitions
    hyperparameter_vect = shuffle(repeat(hyperparameter_vect, n_parameter_iterations))
    exp_list = DDDCSummary[]

    @info "About to run $(length(hyperparameter_vect) ÷ n_parameter_iterations) parameter settings, each $n_parameter_iterations times"

    exp_list_ = @showprogress @distributed for i in 1:size(hyperparameter_vect)
        append!(exp_list_, run_and_extract(hyperparameter_vect[i]))
    end

    folder_name = joinpath(
        "data",
        savename((
            model = "dddc",
            version = version,
            SLURM_ARRAY_JOB_ID = batch_metadata.SLURM_ARRAY_JOB_ID,
            debug = debug,
        )),
        savename((
            start_timestamp = start_timestamp,
            SLURM_ARRAY_TASK_ID = batch_metadata.SLURM_ARRAY_TASK_ID,
        )),
    )
    mkpath(folder_name)
    df = extract_sim_results(exp_list)
    Arrow.write(folder_name * ".arrow", df)
    df = expand_and_extract_dddc(df)
    df_summary = construct_df_summary_dddc(df)
    Arrow.write(folder_name * "_df_summary.arrow", df_summary)
    return exp_list
end
