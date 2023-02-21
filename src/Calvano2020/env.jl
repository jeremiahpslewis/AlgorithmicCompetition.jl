using ReinforcementLearning
using StaticArrays

struct CalvanoEnv <: AbstractEnv
    α::Float64
    β::Float64
    δ::Float64
    n_players::Int8
    memory_length::Int
    price_options::SVector{15, Float64}
    max_iter::Int32
    convergence_threshold::Int32
    n_prices::Int8
    price_index::SVector{15, Int8}
    init_matrix::MMatrix{15, 225, Float32}
    profit_function::Function
    n_state_space::Int
    memory::MMatrix{1, 2Int8}
    is_converged::MVector{2, Bool}
    reward::MVector{2, Float64}
    is_done::MVector{1, Bool}
    p_Bert_nash_equilibrium::Float64
    p_monop_opt::Float64

    function CalvanoEnv(p::CalvanoHyperParameters)
        # Special case starting conditions with 'missing' in lookbacks, think about best way of handling this...
        # TODO: Think about how initial memory should be assigned
        price_options = SVector{15, Float64}(p.price_options)
        n_prices = convert(Int8, length(p.price_options))
        price_index = SVector{15, Int8}(convert.(Int8, 1:n_prices))
        n_players = convert(Int8, p.n_players)
        n_state_space = convert(Int16, n_prices)^(p.memory_length * n_players)
        init_matrix = MMatrix{15, 225, Float32}(zeros(Float32, n_prices, n_state_space))

        new(
            p.α,
            p.β,
            p.δ,
            n_players,
            p.memory_length,
            p.price_options,
            convert(Int32, p.max_iter),
            convert(Int32, p.convergence_threshold),
            n_prices,
            price_index,
            init_matrix,
            p.profit_function,
            n_state_space,
            MMatrix(Int8, 1, 2)(ones(Int8, p.memory_length, p.n_players)), # Memory, note max of 127 prices with Int
            MVector{2, Bool}(fill(false, p.n_players)), # Is converged
            MVector{2, Float64}([0.0, 0.0]), # Reward
            MVector{1, Bool}([false]), # Is done
            p.p_Bert_nash_equilibrium,
            p.p_monop_opt,
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

# map price vector to state
function map_vect_to_int(vect_, base)
    sum(vect_[k]*base^(k-1) for k=1:length(vect_)) # From Julia help / docs
end

function map_int_to_vect(int_val, base, vect_length)
    return digits(Int8, int_val, base=base, pad=vect_length)
end

RLBase.action_space(env::CalvanoEnv, ::Int) = env.price_index # Choice of price

RLBase.action_space(env::CalvanoEnv, ::SimultaneousPlayer) =
    Tuple((i, j) for i in env.price_index for j in env.price_index)

RLBase.legal_action_space(env::CalvanoEnv, p) =
    is_terminated(env) ? () : action_space(env, p)

RLBase.action_space(env::CalvanoEnv) = action_space(env, SIMULTANEOUS_PLAYER)

RLBase.reward(env::CalvanoEnv) = env.is_done[1] ? env.reward : [0, 0]
RLBase.reward(env::CalvanoEnv, p::Int) = reward(env)[p]

RLBase.state_space(env::CalvanoEnv, ::Observation, p) = Base.OneTo(env.n_state_space)

function RLBase.state(env::CalvanoEnv, ::Observation, p)
    map_vect_to_int(env.memory .- 1, env.n_prices) + 1
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
