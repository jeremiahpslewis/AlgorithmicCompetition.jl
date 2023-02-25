using ReinforcementLearning
using StaticArrays

struct AIAPCEnv <: AbstractEnv
    α::Float32
    β::Float32
    δ::Float64
    n_players::Int
    memory_length::Int
    price_options::SVector{15, Float32}
    max_iter::Int
    convergence_threshold::Int
    n_prices::Int
    price_index::SVector{15, Int8}
    init_matrix::MMatrix{15, 225, Float32}
    profit_function::Function
    n_state_space::Int16
    state_space::Base.OneTo{Int}
    state_space_lookup::Array{Int16, 2}
    memory::MVector{2, Int}
    convergence_int::MVector{1, Int}
    is_done::MVector{1, Bool}
    p_Bert_nash_equilibrium::Float32
    p_monop_opt::Float32
    action_space::Tuple
    profit_array::Array{Float32, 3}

    function AIAPCEnv(p::AIAPCHyperParameters)
        # Special case starting conditions with 'missing' in lookbacks, think about best way of handling this...
        # TODO: Think about how initial memory should be assigned
        price_options = SVector{15, Float32}(p.price_options)
        n_prices = length(p.price_options)
        price_index = SVector{15, Int8}(1:n_prices)
        n_players = p.n_players
        n_state_space = n_prices^(p.memory_length * n_players)
        state_space = Base.OneTo(n_state_space)
        init_matrix = MMatrix{15, 225, Float32}(zeros(Float32, n_prices, n_state_space))
        action_space = Tuple((i, j) for i in price_index for j in price_index)

        profit_array = construct_profit_array(action_space, price_options, p.profit_function, n_players)
        state_space_lookup = construct_state_space_lookup(action_space, n_prices)

        new(
            p.α,
            p.β,
            p.δ,
            n_players,
            p.memory_length,
            p.price_options,
            p.max_iter,
            p.convergence_threshold,
            n_prices,
            price_index,
            init_matrix,
            p.profit_function,
            n_state_space,
            state_space,
            state_space_lookup,
            MVector{2, Int}(ones(Int, p.memory_length, p.n_players)), # Memory, note max of 127 prices with Int
            MVector{1, Int}([0]), # Convergence counter
            MVector{1, Bool}([false]), # Is done
            p.p_Bert_nash_equilibrium,
            p.p_monop_opt,
            action_space,
            profit_array,
        )
    end
end

function (env::AIAPCEnv)((p_1, p_2))
    # TODO: Fix support for longer memories
    env.memory .= (p_1, p_2)
    env.is_done[1] = true
end

function construct_state_space_lookup(action_space, n_prices)
    @assert length(action_space) == n_prices^2
    state_space_lookup = zeros(Int16, n_prices, n_prices)
    for (i,j) in action_space
        state_space_lookup[i, j] = map_vect_to_int([i,j], n_prices) - n_prices
    end
    return state_space_lookup
end

# map price vector to state
function map_vect_to_int(vect_, base)
    sum(vect_[k] * base^(k-1) for k=1:length(vect_)) # From Julia help / docs
end

function map_int_to_vect(int_val, base, vect_length)
    return digits(Int, int_val, base=base, pad=vect_length)
end

function construct_profit_array(action_space::NTuple, price_options, profit_function, n_players::Int)
    n_prices = length(price_options)
    # TODO: Carve out into separate function:
    profit_array = zeros(Float32, n_prices, n_prices, n_players)
    for k in 1:n_players
        for (i,j) in action_space
            profit_array[i, j, k] = profit_function([price_options[i], price_options[j]])[k]
        end
    end

    return profit_array
end

RLBase.action_space(env::AIAPCEnv, ::Int) = env.price_index # Choice of price

RLBase.action_space(env::AIAPCEnv, ::SimultaneousPlayer) = env.action_space
    
RLBase.legal_action_space(env::AIAPCEnv, p) =
    is_terminated(env) ? () : action_space(env, p)

RLBase.action_space(env::AIAPCEnv) = action_space(env, SIMULTANEOUS_PLAYER)

function RLBase.reward(env::AIAPCEnv)
    env.is_done[1] ? (@view env.profit_array[env.memory[1], env.memory[2], :]) : SA[0, 0]
end

function RLBase.reward(env::AIAPCEnv, p::Int)
    (@view env.profit_array[env.memory[1], env.memory[2], p])[1]
end

RLBase.state_space(env::AIAPCEnv, ::Observation, p) = env.state_space

function RLBase.state(env::AIAPCEnv, ::Observation, p)
    env.state_space_lookup[env.memory[1], env.memory[2]]
end

RLBase.is_terminated(env::AIAPCEnv) = env.is_done[1]
# TODO: Expand reset function to other params
function RLBase.reset!(env::AIAPCEnv)
    env.is_done[1] = false
end

RLBase.players(::AIAPCEnv) = (1, 2)
RLBase.current_player(::AIAPCEnv) = SIMULTANEOUS_PLAYER
RLBase.NumAgentStyle(::AIAPCEnv) = MultiAgent(2)
RLBase.DynamicStyle(::AIAPCEnv) = SIMULTANEOUS
RLBase.ActionStyle(::AIAPCEnv) = MINIMAL_ACTION_SET
RLBase.InformationStyle(::AIAPCEnv) = IMPERFECT_INFORMATION
RLBase.StateStyle(::AIAPCEnv) = Observation{Int16}()
RLBase.RewardStyle(::AIAPCEnv) = STEP_REWARD
RLBase.UtilityStyle(::AIAPCEnv) = GENERAL_SUM
RLBase.ChanceStyle(::AIAPCEnv) = DETERMINISTIC
