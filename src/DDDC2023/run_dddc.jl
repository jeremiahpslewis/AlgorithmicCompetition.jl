using Distributed
using Random
using StatsBase
using Arrow
using ProgressMeter

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
    precompile = false,
    state_space_tremble_parameters = [0.0],
    action_space_tremble_parameters = [0.0],
    write_to_file_return_none = true,
)
    signal_quality_vect = [[true, false]] # With signal_quality_range over both weak and strong, [false, false] case is redundant

    frequency_high_demand_range = [0, 0.5, 1.0]

    # if n_grid_increments == 1, then only consider perfect and noisy signal cases
    @assert n_grid_increments >= 0 "n_grid_increments must be greater than or equal to 0, is $n_grid_increments"
    n_grid_increments = n_grid_increments == 0 ? 1 : n_grid_increments
    @info "Running DDDC with n_grid_increments = $n_grid_increments"
    signal_quality_level_range = Float64.(range(0.5, 1.0, n_grid_increments + 1))

    @info "Signal quality level range: $signal_quality_level_range"

    competition_params_dict = Dict(
        :low => CompetitionParameters(0.25, 0.25, (2, 2), (1, 1)),
        :high => CompetitionParameters(0.25, -0.25, (2, 2), (1, 1)), # Parameter values aligned with Calvano 2020 Stochastic Demand case
    )

    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

    α = Float64(0.15)
    β = Float64(4e-1)
    δ = 0.95

    # Loop over state tremble with constant action tremble = 0.0
    data_demand_digital_param_set_state = [
        DDDCExperimentalParams(
            weak_signal_quality_level = weak_signal_quality_level,
            strong_signal_quality_level = strong_signal_quality_level,
            signal_is_strong = signal_quality_players,
            frequency_high_demand = frequency_high_demand,
            state_space_tremble_frequency = state_tremble,
            action_space_tremble_frequency = action_space_tremble_frequency
        ) for state_tremble in state_space_tremble_parameters for
        frequency_high_demand in frequency_high_demand_range for
        signal_quality_players in signal_quality_vect for
        weak_signal_quality_level in signal_quality_level_range for
        strong_signal_quality_level in signal_quality_level_range if weak_signal_quality_level <= strong_signal_quality_level for
        action_space_tremble_frequency in [0.0, 0.01]
    ]

    # Loop over action tremble with constant state tremble = 0.0
    data_demand_digital_param_set_action = [
        DDDCExperimentalParams(
            weak_signal_quality_level = weak_signal_quality_level,
            strong_signal_quality_level = strong_signal_quality_level,
            signal_is_strong = signal_quality_players,
            frequency_high_demand = frequency_high_demand,
            state_space_tremble_frequency = state_space_tremble_frequency,
            action_space_tremble_frequency = action_tremble
        ) for action_tremble in action_space_tremble_parameters for
        frequency_high_demand in frequency_high_demand_range for
        signal_quality_players in signal_quality_vect for
        weak_signal_quality_level in signal_quality_level_range for
        strong_signal_quality_level in signal_quality_level_range if weak_signal_quality_level <= strong_signal_quality_level for
        state_space_tremble_frequency in [0.0, 0.01]
    ]

    data_demand_digital_param_set = [
        data_demand_digital_param_set_state...,
        data_demand_digital_param_set_action...
    ]

    # Always run 'missing' signal stochastic demand case, 0.0
    missing_signal_level = 0.0
    data_demand_digital_param_set_missing_signal_state = [
        DDDCExperimentalParams(
            weak_signal_quality_level = missing_signal_level,
            strong_signal_quality_level = active_signal_quality_level,
            signal_is_strong = signal_quality_players,
            frequency_high_demand = frequency_high_demand,
            state_space_tremble_frequency = state_tremble,
            action_space_tremble_frequency = action_space_tremble_frequency
        ) for state_tremble in state_space_tremble_parameters for
        frequency_high_demand in frequency_high_demand_range for
        signal_quality_players in signal_quality_vect for
        active_signal_quality_level in [signal_quality_level_range..., missing_signal_level] for
        action_space_tremble_frequency in [0.0, 0.01]
    ]

    data_demand_digital_param_set_missing_signal_action = [
        DDDCExperimentalParams(
            weak_signal_quality_level = missing_signal_level,
            strong_signal_quality_level = active_signal_quality_level,
            signal_is_strong = signal_quality_players,
            frequency_high_demand = frequency_high_demand,
            state_space_tremble_frequency = state_space_tremble_frequency,
            action_space_tremble_frequency = action_tremble
        ) for action_tremble in action_space_tremble_parameters for
        frequency_high_demand in frequency_high_demand_range for
        signal_quality_players in signal_quality_vect for
        active_signal_quality_level in [signal_quality_level_range..., missing_signal_level]
        for state_space_tremble_frequency in [0.0, 0.01]
    ]

    data_demand_digital_param_set_missing_signal = [
        data_demand_digital_param_set_missing_signal_state...,
        data_demand_digital_param_set_missing_signal_action...
    ]

    # Always run 'sunspot' joint random signal stochastic demand case -1.0
    signal_quality_joint_vect = [-1.0]
    data_demand_digital_param_special_set_state = [
        DDDCExperimentalParams(
            weak_signal_quality_level = signal_quality_level,
            strong_signal_quality_level = signal_quality_level,
            signal_is_strong = signal_quality_players,
            frequency_high_demand = frequency_high_demand,
            state_space_tremble_frequency = state_tremble,
            action_space_tremble_frequency = action_space_tremble_frequency
        ) for state_tremble in state_space_tremble_parameters for
        frequency_high_demand in frequency_high_demand_range for
        signal_quality_players in signal_quality_vect for
        signal_quality_level in signal_quality_joint_vect for
        action_space_tremble_frequency in [0.0, 0.01]
    ]

    data_demand_digital_param_special_set_action = [
        DDDCExperimentalParams(
            weak_signal_quality_level = signal_quality_level,
            strong_signal_quality_level = signal_quality_level,
            signal_is_strong = signal_quality_players,
            frequency_high_demand = frequency_high_demand,
            state_space_tremble_frequency = state_space_tremble_frequency,
            action_space_tremble_frequency = action_tremble
        ) for action_tremble in action_space_tremble_parameters for
        frequency_high_demand in frequency_high_demand_range for
        signal_quality_players in signal_quality_vect for
        signal_quality_level in signal_quality_joint_vect for
        state_space_tremble_frequency in [0.0, 0.01]
    ]

    data_demand_digital_param_special_set = [
        data_demand_digital_param_special_set_state...,
        data_demand_digital_param_special_set_action...
    ]

    data_demand_digital_param_set = [
        data_demand_digital_param_set...,
        data_demand_digital_param_set_missing_signal...,
        data_demand_digital_param_special_set...,
    ]

    hyperparameter_vect = [
        DDDCHyperParameters(
            α,
            β, # \beta value
            δ,
            max_iter,
            competition_solution_dict,
            data_demand_digital_params;
            convergence_threshold = convergence_threshold,
        ) for data_demand_digital_params in data_demand_digital_param_set
    ]

    # Shuffle hyperparameter_vect, extend according to number of repetitions
    hyperparameter_vect = shuffle(repeat(hyperparameter_vect, n_parameter_iterations))

    if debug
        hyperparameter_vect = hyperparameter_vect[1:10:end]
    end

    @info "About to run $(length(hyperparameter_vect) ÷ n_parameter_iterations) parameter settings, each $n_parameter_iterations times"

    write_to_file_path = mktempdir()

    @showprogress @distributed for hyperparam_item in hyperparameter_vect
        nothing = run_and_extract(hyperparam_item, write_to_file_path=write_to_file_path, write_to_file_return_none=write_to_file_return_none)
    end

    if !write_to_file_return_none
        return exp_output
    end
    @info "run_and_extract completed"

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
    
    
    single_run_files = readdir(write_to_file_path, join = true)
    single_run_dfs = read_raw_arrow_file.(single_run_files)
    all_run_df = vcat(single_run_dfs...)

    if !precompile
        @info "Saving $(nrow(all_run_df)) results to $folder_name"
        Arrow.write(folder_name * ".arrow", all_run_df)
    end

    @info "Extracting summary"

    df = expand_and_extract_dddc(all_run_df)
    df_summary = construct_df_summary_dddc(df)

    @info "Saving summary to $folder_name"
    if !precompile
        Arrow.write(folder_name * "_df_summary.arrow", df_summary)
    end
    return df_summary
end
