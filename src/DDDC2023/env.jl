using ReinforcementLearningCore
using ReinforcementLearningBase
using StaticArrays

const player_lookup = (; Symbol(1) => 1, Symbol(2) => 2)
const demand_lookup = (; :high => 1, :low => 2)
const player_to_index = (; Symbol(1) => 1, Symbol(2) => 2)
const demand_to_index = (; :high => 1, :low => 2)

"""
    DDDCEnv(p::AIAPCHyperParameters)

    Build an environment to reproduce the results of the Lewis 2023 

    TODO: Add citation here.
"""
struct DDDCEnv <: AbstractEnv # N is profit_array dimension
    α::Float64                              # Learning parameter
    β::Float64                              # Exploration parameter
    δ::Float64                              # Discount factor
    max_iter::Int                           # Maximum number of iterations
    convergence_threshold::Int              # Convergence threshold

    n_players::Int                          # Number of players
    price_options::Vector{Float64}      # Price options
    price_index::Vector{Int8}           # Price indices

    competition_params_dict::Dict{Symbol, CompetitionParameters} # Competition parameters, true = high, false = low
    memory::Vector{CartesianIndex{2}}       # Memory vector (previous prices)
    is_high_demand_signals::Vector{Bool}    # [true, false] if demand signal is high for player one and low for player two for a given episode
    prev_is_high_demand_signals::Vector{Bool}    # [true, false] if demand signal is high for player one and low for player two for a given episode
    is_high_demand_episode::Vector{Bool}    # [true] if demand is high for a given episode
    state_space::Base.OneTo{Int16}          # State space
    state_space_lookup::Array{Int16, 4}       # State space lookup table

    n_prices::Int                           # Number of price options
    n_state_space::Int64                    # Number of states

    convergence_dict::Dict{Symbol,Bool}     # Convergence status for each player
    is_done::MVector{1,Bool}                # Episode is complete

    p_Bert_nash_equilibrium::Dict{Symbol,Float64}        # Nash equilibrium prices for low and high demand (Betrand price)
    p_monop_opt::Dict{Symbol,Float64}                    # Monopoly optimal prices for low and high demand

    action_space::Tuple                     # Action space
    profit_array::Array{Float64,4}          # Profit given price pair as coordinates

    data_demand_digital_params::DataDemandDigitalParams # Parameters for Data/Demand/Digital AIAPC extension

    function DDDCEnv(p::DDDCHyperParameters)
        price_options = Vector{Float64}(p.price_options)
        n_prices = length(p.price_options)
        price_index = Vector{Int8}(Int8.(1:n_prices))
        n_players = p.n_players
        n_state_space = 4 * n_prices^(p.memory_length * n_players) # 2^2 = 4 possible demand states (ground truth and signal)

        state_space = Base.OneTo(Int16(n_state_space))
        action_space = construct_DDDC_action_space(price_index)
        profit_array =
            construct_DDDC_profit_array(price_options, p.competition_params_dict, n_players)
        state_space_lookup = construct_DDDC_state_space_lookup(action_space, n_prices)
        is_high_demand_episode = rand(Bool, 1)

        new(
            p.α,
            p.β,
            p.δ,
            p.max_iter,
            p.convergence_threshold,
            n_players,
            p.price_options,
            price_index,
            p.competition_params_dict,
            initialize_price_memory(price_index, p.n_players), # Memory, randomly initialized
            get_demand_signals(p.data_demand_digital_params, is_high_demand_episode[1]), # Current demand, randomly initialized
            get_demand_signals(p.data_demand_digital_params, is_high_demand_episode[1]), # Previous demand, randomly initialized
            is_high_demand_episode,
            state_space,
            state_space_lookup,
            n_prices,
            n_state_space,
            Dict(Symbol(1) => false, Symbol(2) => false), # Convergence dict
            MVector{1,Bool}([false]), # Episode is done indicator
            p.p_Bert_nash_equilibrium,
            p.p_monop_opt,
            action_space,
            profit_array,
            p.data_demand_digital_params,
        )
    end
end

"""
    RLBase.act!(env::DDDCEnv, price_tuple::Tuple{Int8,Int8})

Act in the environment by setting the memory to the given price tuple and setting `is_done` to `true`.
"""
function RLBase.act!(env::DDDCEnv, price_tuple::CartesianIndex{2})
    # TODO: Fix support for longer memories
    env.memory[1] = price_tuple
    env.is_done[1] = true
end

RLBase.action_space(env::DDDCEnv, ::Symbol) = env.price_index # Choice of price

RLBase.action_space(env::DDDCEnv, ::SimultaneousPlayer) = env.action_space

RLBase.legal_action_space(env::DDDCEnv, p) = is_terminated(env) ? () : action_space(env, p)

const legal_action_space_mask_object_DDDC = [Int8.(1:15)...]

RLBase.legal_action_space_mask(env::DDDCEnv, player::Symbol) =
legal_action_space_mask_object_DDDC

RLBase.action_space(env::DDDCEnv) = action_space(env, SIMULTANEOUS_PLAYER)

const zero_tuple = Tuple{Float64,Float64}([0, 0])

"""
    RLBase.reward(env::DDDCEnv)

Return the reward for the current state. If the episode is done, return the profit, else return `(0, 0)`.
"""
function RLBase.reward(env::DDDCEnv)
    memory_index = env.memory[1]
    env.is_done[1] ? Tuple{Float64,Float64}(env.profit_array[memory_index, :, demand_to_index[env.demand_signal]]) : zero_tuple
end

"""
    RLBase.reward(env::DDDCEnv, p::Int)

Return the reward for the current state for player `p` as an integer. If the episode is done, return the profit, else return `0`.
"""
function RLBase.reward(env::DDDCEnv, p::Int)::Float64
    profit_array = env.profit_array
    memory_index = env.memory[1]
    return _reward(
        profit_array,
        memory_index,
        env.is_high_demand_episode[1],
        p
        )
end

function _reward(profit::Array{Float64,4},
    memory_index::CartesianIndex{2},
    is_high_demand_episode::Bool,
    p::Int)::Float64
    if is_high_demand_episode
        demand_index_ = 2
    else
        demand_index_ = 1
    end

    return profit[memory_index, p, demand_index_]
end

"""
    RLBase.reward(env::DDDCEnv, p::Int)

Return the reward for the current state for player `p` as a symbol. If the episode is done, return the profit, else return `0`.
"""
RLBase.reward(env::DDDCEnv, p::Symbol) = reward(env, player_lookup[p])

RLBase.state_space(env::DDDCEnv, ::Observation, p) = env.state_space

# State without player spec is a noop
RLBase.state(env::DDDCEnv) = nothing

"""
    RLBase.state(env::DDDCEnv, player::Symbol)

Return the current state as an integer, mapped from the environment memory.
"""
function RLBase.state(env::DDDCEnv, p::Symbol)
    memory_index = env.memory[1]
    # State is defined by memory, as in AIAPC, plus demand signal given to a player
    index_ = player_to_index[p]
    _is_high_demand_signal = env.is_high_demand_signals[index_]
    _demand_signal = _is_high_demand_signal ? :high : :low
    demand_signal_index = demand_to_index[_demand_signal]
    _prev_is_high_demand_signal = env.prev_is_high_demand_signals[index_]
    _prev_demand_signal = _prev_is_high_demand_signal ? :high : :low
    prev_demand_signal_index = demand_to_index[_prev_demand_signal]

    # State space is indexed by: memory (price x price, length 2), current demand signal, previous demand signal
    env.state_space_lookup[memory_index, demand_signal_index, prev_demand_signal_index]
end

"""
    RLBase.is_terminated(env::DDDCEnv)

Return whether the episode is done.
"""
RLBase.is_terminated(env::DDDCEnv) = env.is_done[1]


function RLBase.reset!(env::DDDCEnv)
    env.is_done[1] = false

    # Determine whether next episode is a high demand episode
    is_high_demand_episode = get_demand_level(env.data_demand_digital_params)

    # Update demand signals
    env.prev_is_high_demand_signals .= env.is_high_demand_signals
    env.is_high_demand_signals .= get_demand_signals(env.data_demand_digital_params, is_high_demand_episode)

    # Update demand level
    env.is_high_demand_episode[1] = is_high_demand_episode
end

RLBase.players(::DDDCEnv) = (Symbol(1), Symbol(2))
RLBase.current_player(::DDDCEnv) = SIMULTANEOUS_PLAYER
RLBase.NumAgentStyle(::DDDCEnv) = MultiAgent(2)
RLBase.DynamicStyle(::DDDCEnv) = SIMULTANEOUS
RLBase.ActionStyle(::DDDCEnv) = MINIMAL_ACTION_SET
RLBase.InformationStyle(::DDDCEnv) = IMPERFECT_INFORMATION
RLBase.StateStyle(::DDDCEnv) = Observation{Int64}()
RLBase.RewardStyle(::DDDCEnv) = STEP_REWARD
RLBase.UtilityStyle(::DDDCEnv) = GENERAL_SUM
RLBase.ChanceStyle(::DDDCEnv) = DETERMINISTIC

function RLBase.plan!(explorer::Ex, learner::L, env::DDDCEnv, player::Symbol) where {Ex<:AbstractExplorer,L<:AbstractLearner}
    legal_action_space_ = RLBase.legal_action_space_mask(env, player)
    return RLBase.plan!(explorer, RLCore.forward(learner, state(env, player)), legal_action_space_)
end
