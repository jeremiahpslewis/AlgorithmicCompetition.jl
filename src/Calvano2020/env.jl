using ReinforcementLearning
using StaticArrays

struct CalvanoEnv <: AbstractEnv
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
    memory::MVector{2, Int}
    convergence_int::MVector{1, Int}
    is_done::MVector{1, Bool}
    p_Bert_nash_equilibrium::Float32
    p_monop_opt::Float32
    action_space::Tuple
    profit_array::Array{Float32, 3}

    function CalvanoEnv(p::CalvanoHyperParameters)
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

        # TODO: Carve out into separate function:
        profit_array = zeros(Float32, n_prices, n_prices, n_players)
        for k in 1:n_players
            for (i,j) in action_space
                profit_array[i, j, k] = p.profit_function([price_options[i], price_options[j]])[k]
            end
        end

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

function (env::CalvanoEnv)((p_1, p_2))
    # TODO: Fix support for longer memories
    env.memory .= (p_1, p_2)
    env.is_done[1] = true
end

# map price vector to state
function map_vect_to_int(vect_, base)
    sum(vect_[k] * base^(k-1) for k=1:length(vect_)) # From Julia help / docs
end

function map_int_to_vect(int_val, base, vect_length)
    return digits(Int, int_val, base=base, pad=vect_length)
end

RLBase.action_space(env::CalvanoEnv, ::Int) = env.price_index # Choice of price

RLBase.action_space(env::CalvanoEnv, ::SimultaneousPlayer) = env.action_space
    
RLBase.legal_action_space(env::CalvanoEnv, p) =
    is_terminated(env) ? () : action_space(env, p)

RLBase.action_space(env::CalvanoEnv) = action_space(env, SIMULTANEOUS_PLAYER)

function RLBase.reward(env::CalvanoEnv)
    env.is_done[1] ? (@view env.profit_array[env.memory[1], env.memory[2], :]) : SA[0, 0]
end

function RLBase.reward(env::CalvanoEnv, p::Int)
     env.is_done[1] ? env.profit_array[env.memory[1], env.memory[2], p] : 0
end

RLBase.state_space(env::CalvanoEnv, ::Observation, p) = env.state_space

function RLBase.state(env::CalvanoEnv, ::Observation, p)
    map_vect_to_int(env.memory .- 1, env.n_prices) + 1
end

RLBase.is_terminated(env::CalvanoEnv) = env.is_done[1]
# TODO: Expand reset function to other params
function RLBase.reset!(env::CalvanoEnv)
    env.is_done[1] = false
end

RLBase.players(::CalvanoEnv) = (1, 2)
RLBase.current_player(::CalvanoEnv) = SIMULTANEOUS_PLAYER
RLBase.NumAgentStyle(::CalvanoEnv) = MultiAgent(2)
RLBase.DynamicStyle(::CalvanoEnv) = SIMULTANEOUS
RLBase.ActionStyle(::CalvanoEnv) = MINIMAL_ACTION_SET
RLBase.InformationStyle(::CalvanoEnv) = IMPERFECT_INFORMATION
RLBase.StateStyle(::CalvanoEnv) = Observation{Int16}()
RLBase.RewardStyle(::CalvanoEnv) = STEP_REWARD
RLBase.UtilityStyle(::CalvanoEnv) = GENERAL_SUM
RLBase.ChanceStyle(::CalvanoEnv) = DETERMINISTIC
