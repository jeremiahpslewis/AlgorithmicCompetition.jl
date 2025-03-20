import Base.push!
using ReinforcementLearning
using DrWatson
using Logging
using LoggingExtras

const player_to_index = Dict(Player(1) => 1, Player(2) => 2)
const demand_to_index = (; :high => 1, :low => 2)

# Handle CartesianIndex actions
function Base.push!(
    multiagent::MultiAgentPolicy,
    ::PostActStage,
    env::E,
    actions::CartesianIndex,
) where {E<:AbstractEnv}
    actions = Tuple(actions)
    Base.push!(multiagent, PostActStage(), env, actions)
end

# Fixed Calvano 2020 ν function, source: https://kasberger.github.io/assets/pdf/algorithmic%20cooperation%20online%20appendix.pdf
# Number of visits to a state-action pair given (m = action space cardinality, n = n players, k = memory periods, β) in round k in Q-learning with ε-greedy exploration
ν(m::Int64, n::Int64, k::Int64, β::Float64) =
    (m - 1)^n / (m^(k * n + n + 1) * (1 - exp(-β * (n + 1))))
function ν_inverse(m::Int64, n::Int64, k::Int64, ν_target::Float64)
    num = (m - 1)^n
    den = m^(k * n + n + 1) * ν_target
    β = -log(1 - num / den) / (n + 1)
    return β
end


# @test ν(15, 2, 1, ν_inverse(15, 2, 1, 20)) ≈ 20

# ν(15, 2, 1, Float64(2e-6))
# For random signals, exploration is reduced by factor of signal cardinality (as the signals are random and uncorrelated) in expectation.
# But! The likelihood of exploring a given state is not evenly distributed across states, so this is a simplification.
# If the minimum state exploration is to be controlled, ν needs to be adjusted by the lower frequency signal probability squared. 

# Frequency with which the least frequent signal realization is observed
function frequency_least_frequent_signal_value(
    frequency_high_demand::Float64,
    weak_signal_quality_level::Float64,
)
    p_ =
        frequency_high_demand * weak_signal_quality_level +
        (1 - frequency_high_demand) * (1 - weak_signal_quality_level)
    return min(p_, 1 - p_)
end

# @test frequency_least_frequent_signal(0.8, 0.9) == 0.2599999999999999

# Correct for the fact that the least frequent signal is observed less frequently
# Adjust 'visit' metric ν for the fact that the demand signals distribute 'visits' across signal space, and do so unevenly
function ν_tilde(ν_, frequency_high_demand, weak_signal_quality_level)
    signal_distribution_correction =
        frequency_least_frequent_signal_value(
            frequency_high_demand,
            weak_signal_quality_level,
        )^2
    return ν_ / signal_distribution_correction
end

# @test ν_inverse(15, 2, 1, ν_tilde(ν_, 0.5, 0.5)) ≈ 1.0754475510580863e-6

# @test ν_inverse(15, 2, 1, ν_tilde(ν_, 0.9, 0.1)) ≈ ν_inverse(15, 2, 1, ν_tilde(ν_, 0.1, 0.9))

function extract_params_from_environment()
    @info "Extracting parameters from environment variables."

    version = get(ENV, "VERSION", "v1")
    debug = parse(Int, get(ENV, "DEBUG", "1")) == 1
    if haskey(ENV, "SLURM_ARRAY_TASK_ID")
        SLURM_ARRAY_TASK_ID = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "1"))
        SLURM_ARRAY_JOB_ID = parse(Int, get(ENV, "SLURM_ARRAY_JOB_ID", "1"))
    else
        SLURM_ARRAY_TASK_ID = 1
        SLURM_ARRAY_JOB_ID = parse(Int, get(ENV, "SLURM_JOB_ID", "1"))
    end
    log_dir = get(ENV, "LOG_DIR", "log")
    log_path = joinpath(log_dir, "$(SLURM_ARRAY_JOB_ID)_$(SLURM_ARRAY_TASK_ID).log")
    n_cores = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "2"))
    n_grid_increments = parse(Int, get(ENV, "N_GRID_INCREMENTS", "10"))
    n_parameter_iterations = parse(Int, get(ENV, "N_PARAMETER_ITERATIONS", "1"))

    params = Dict(
        :version => version,
        :debug => debug,
        :log_path => log_path,
        :SLURM_ARRAY_TASK_ID => SLURM_ARRAY_TASK_ID,
        :SLURM_ARRAY_JOB_ID => SLURM_ARRAY_JOB_ID,
        :n_cores => n_cores,
        :n_grid_increments => n_grid_increments,
        :n_parameter_iterations => n_parameter_iterations,
    )

    # Overrride in case of debugging
    if params[:debug] && Sys.isapple()
        params[:max_iter] = Int(1e9)
        params[:convergence_threshold] = Int(1e5)
    elseif params[:debug]
        params[:n_grid_increments] = 10
        params[:max_iter] = Int(1e6)
        params[:convergence_threshold] = Int(1e2)
    else
        params[:max_iter] = Int(1e9)
        params[:convergence_threshold] = Int(1e5)
    end
    @info "Parameters extracted from environment variables: $(params)"
    return params
end

function setup_logger(params)
    mkpath(dirname(params[:log_path]))
    f_logger = FileLogger(params[:log_path]; append = true)
    debuglogger = MinLevelLogger(f_logger, Logging.Info)
    global_logger(debuglogger)
    @info "Logger setup complete."
end
