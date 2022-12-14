using ReinforcementLearning
using StaticArrays

struct CalvanoEnv <: AbstractEnv
    α::Float64
    β::Float64
    δ::Float64
    n_players::UInt8
    memory_length::UInt8
    price_options::Vector{Float64}
    max_iter::Int
    convergence_threshold::Int
    n_prices::UInt8
    price_index::Vector{Int}
    convergence_check::ConvergenceCheck
    init_matrix::Matrix{Float32}
    profit_function::Function
    n_state_space::UInt16
    memory::Matrix{UInt16}
    is_converged::Vector{Bool}
    reward::Vector{Float64}
    is_done::Vector
    p_Bert_nash_equilibrium::Float64
    p_monop_opt::Float64

    function CalvanoEnv(
        α::Float64,
        β::Float64,
        δ::Float64,
        n_players::Int,
        memory_length::Int,
        price_options::Vector{Float64},
        max_iter::Int,
        convergence_threshold::Int,
        profit_function,
        p_Bert_nash_equilibrium::Float64,
        p_monop_opt::Float64,
    )
        # Special case starting conditions with 'missing' in lookbacks, think about best way of handling this...
        # TODO: Think about how initial memory should be assigned
        n_prices = length(price_options)
        price_index = 1:n_prices
        n_state_space = n_prices^(memory_length * n_players)
        convergence_check =
            ConvergenceCheck(n_state_space, n_players)
        init_matrix = zeros(Float32, n_prices, n_state_space)

        new(
            α,
            β,
            δ,
            n_players,
            convert(UInt8, memory_length),
            price_options,
            max_iter,
            convergence_threshold,
            convert(UInt32, n_prices),
            price_index,
            convergence_check,
            init_matrix,
            profit_function,
            convert(UInt16, n_state_space),
            ones(UInt16, memory_length, n_players), # Memory, note max of 256 prices with UInt8
            fill(false, n_players), # Is converged
            [0.0, 0.0], # Reward
            [false], # Is done
            p_Bert_nash_equilibrium,
            p_monop_opt,
        )
    end
end

function (env::CalvanoEnv)((p_1, p_2))
    # Convert from price indices to price level, compute profit
    env.reward .= env.profit_function([env.price_options[p_1], env.price_options[p_2]])

    env.memory .= circshift(env.memory, -1)
    env.memory[end, :] .= [p_1, p_2]

    env.is_done[1] = true
end

function map_memory_to_state(v, n_prices)
    v = vec(v)
    sum((v .- 1) .* n_prices .^ ((1:length(v)) .- 1)) + 1
end

RLBase.action_space(env::CalvanoEnv, ::Int) = env.price_index # Choice of price

RLBase.action_space(::CalvanoEnv, ::SimultaneousPlayer) =
    Tuple((i, j) for i in env.price_index for j in env.price_index)

RLBase.legal_action_space(env::CalvanoEnv, p) =
    is_terminated(env) ? () : action_space(env, p)

RLBase.action_space(env::CalvanoEnv) = action_space(env, SIMULTANEOUS_PLAYER)

RLBase.reward(env::CalvanoEnv) = env.is_done[1] ? env.reward : [0, 0]
RLBase.reward(env::CalvanoEnv, p::Int) = reward(env)[p]

RLBase.state_space(::CalvanoEnv, ::Observation, p) = Base.OneTo(env.n_state_space)

function RLBase.state(env::CalvanoEnv, ::Observation, p)
    map_memory_to_state(env.memory, env.n_prices)
end

RLBase.is_terminated(env::CalvanoEnv) = env.is_done[1]
# TODO: Expand reset function to other params
RLBase.reset!(env::CalvanoEnv) = env.is_done[1] = false
RLBase.players(::CalvanoEnv) = (1, 2)
RLBase.current_player(::CalvanoEnv) = SIMULTANEOUS_PLAYER
RLBase.NumAgentStyle(::CalvanoEnv) = MultiAgent(2)
RLBase.DynamicStyle(::CalvanoEnv) = SIMULTANEOUS
RLBase.ActionStyle(::CalvanoEnv) = MINIMAL_ACTION_SET
RLBase.InformationStyle(::CalvanoEnv) = IMPERFECT_INFORMATION
RLBase.StateStyle(::CalvanoEnv) = Observation{Int}()
RLBase.RewardStyle(::CalvanoEnv) = STEP_REWARD
RLBase.UtilityStyle(::CalvanoEnv) = GENERAL_SUM
RLBase.ChanceStyle(::CalvanoEnv) = DETERMINISTIC
