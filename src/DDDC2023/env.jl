using ReinforcementLearning

const player_to_index = Dict(Player(1) => 1, Player(2) => 2)
const demand_to_index = (; :high => 1, :low => 2)

mutable struct DDDCMemory
    prices::CartesianIndex{2}
    signals::Vector{Bool}
    demand_state::Symbol
    reward::Vector{Float64} # NOTE: Not strictly part of the state-defining memory, but used to store reward for each player
end

"""
    DDDCEnv(p::AIAPCHyperParameters)

    Build an environment to reproduce the results of the Lewis 2023 extentions to AIAPC.
"""
struct DDDCEnv <: AbstractEnv # N is profit_array dimension
    α::Float64                              # Learning parameter
    β::Float64                              # Exploration parameter
    δ::Float64                              # Discount factor
    max_iter::Int                           # Maximum number of iterations
    convergence_threshold::Int              # Convergence threshold

    n_players::Int                          # Number of players
    price_options::Vector{Float64}          # Price options
    price_index::Vector{Int64}               # Price indices

    competition_params_dict::Dict{Symbol,CompetitionParameters} # Competition parameters, true = high, false = low
    memory::DDDCMemory                      # Memory struct (previous prices, signals, demand state)
    is_high_demand_signals::Vector{Bool}    # [true, false] if demand signal is high for player one and low for player two for a given episode
    is_high_demand_episode::Vector{Bool}    # [true] if demand is high for a given episode
    state_space::Base.OneTo{Int64}          # State space
    state_space_lookup::Array{Int64,4}      # State space lookup table

    n_prices::Int                           # Number of price options
    n_state_space::Int64                    # Number of states

    convergence_vect::Vector{Bool}          # Convergence status for each player
    is_done::Vector{Bool}                   # Episode is complete

    p_Bert_nash_equilibrium::Dict{Symbol,Float64} # Nash equilibrium prices for low and high demand (Betrand price)
    p_monop_opt::Dict{Symbol,Float64}       # Monopoly optimal prices for low and high demand

    action_space::Tuple                     # Action space
    profit_array::Array{Float64,4}          # Profit given price pair as coordinates

    data_demand_digital_params::DataDemandDigitalParams # Parameters for Data/Demand/Digital AIAPC extension

    reward::Vector{Float64}

    function DDDCEnv(p::DDDCHyperParameters)
        price_options = Vector{Float64}(p.price_options)
        n_prices = length(p.price_options)
        price_index = Vector{Int64}(Int64.(1:n_prices))
        n_players = p.n_players
        n_state_space = 4 * n_prices^(p.memory_length * n_players) # 2^2 = 4 possible demand states (ground truth and signal)

        state_space = Base.OneTo(Int64(n_state_space))
        action_space = construct_DDDC_action_space(price_index)
        profit_array =
            construct_DDDC_profit_array(price_options, p.competition_params_dict, n_players)
        state_space_lookup = construct_DDDC_state_space_lookup(action_space, n_prices)

        is_high_demand_prev_episode = rand(Bool)
        is_high_demand_episode = rand(Bool)

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
            DDDCMemory( # Memory, randomly initialized
                initialize_price_memory(price_index, p.n_players),
                get_demand_signals(
                    p.data_demand_digital_params,
                    is_high_demand_prev_episode,
                ),
                is_high_demand_prev_episode ? :high : :low,
                [0.0, 0.0],
            ),
            get_demand_signals(p.data_demand_digital_params, is_high_demand_episode), # Current demand, randomly initialized
            Bool[is_high_demand_episode],
            state_space,
            state_space_lookup,
            n_prices,
            n_state_space,
            Bool[false, false], # Convergence vector
            Bool[false], # Episode is done indicator
            p.p_Bert_nash_equilibrium,
            p.p_monop_opt,
            action_space,
            profit_array,
            p.data_demand_digital_params,
            Float64[0.0, 0.0],
        )
    end
end

"""
    RLBase.act!(env::DDDCEnv, price_tuple::Tuple{Int64,Int64})

Act in the environment by setting the memory to the given price tuple and setting `is_done` to `true`.
"""
function RLBase.act!(env::DDDCEnv, price_tuple::CartesianIndex{2})
    # TODO: Fix support for longer memories
    demand_state = env.is_high_demand_episode[1] ? :high : :low

    # Reward is based on prices chosen & demand state
    env.memory.reward .= env.profit_array[price_tuple, :, demand_to_index[demand_state]]

    # Update 'memory' data for next episode
    env.memory.prices = price_tuple
    env.memory.signals = copy(env.is_high_demand_signals)
    env.memory.demand_state = demand_state

    # Determine whether next episode is a high demand episode and update
    env.is_high_demand_episode[1] = get_demand_level(env.data_demand_digital_params)

    # Update demand signals
    env.is_high_demand_signals .=
        get_demand_signals(env.data_demand_digital_params, env.is_high_demand_episode[1])
    env.is_done[1] = true
end

RLBase.action_space(env::DDDCEnv, ::Symbol) = env.price_index # Choice of price

RLBase.action_space(env::DDDCEnv, ::SimultaneousPlayer) = env.action_space

RLBase.legal_action_space(env::DDDCEnv, p) = is_terminated(env) ? () : action_space(env, p)

const legal_action_space_mask_object_DDDC = fill(true, 15)

RLBase.legal_action_space_mask(env::DDDCEnv, player::Symbol) =
    legal_action_space_mask_object_DDDC

RLBase.action_space(env::DDDCEnv) = action_space(env, SIMULTANEOUS_PLAYER)


RLBase.reward(env::DDDCEnv, p::Symbol) =
    env.is_done[1] ? env.memory.reward[player_to_index[p]] : zero(Float64)

RLBase.state_space(env::DDDCEnv, ::Observation, p) = env.state_space

# State without player spec is a noop
RLBase.state(env::DDDCEnv) = nothing

"""
    RLBase.state(env::DDDCEnv, player::Symbol)

Return the current state as an integer, mapped from the environment memory.
"""
function RLBase.state(env::DDDCEnv, p::Symbol)
    memory_index = env.memory.prices
    # State is defined by memory, as in AIAPC, plus demand signal given to a player
    index_ = player_to_index[p]

    _is_high_demand_signal = env.is_high_demand_signals[index_]
    _demand_signal = _is_high_demand_signal ? :high : :low
    demand_signal_index = demand_to_index[_demand_signal]

    _prev_is_high_demand_signal = env.memory.signals[index_]
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
end

RLBase.players(::DDDCEnv) = (Player(1), Player(2))
RLBase.current_player(::DDDCEnv) = SIMULTANEOUS_PLAYER
RLBase.NumAgentStyle(::DDDCEnv) = MultiAgent(2)
RLBase.DynamicStyle(::DDDCEnv) = SIMULTANEOUS
RLBase.ActionStyle(::DDDCEnv) = MINIMAL_ACTION_SET
RLBase.InformationStyle(::DDDCEnv) = IMPERFECT_INFORMATION
RLBase.StateStyle(::DDDCEnv) = Observation{Int64}()
RLBase.RewardStyle(::DDDCEnv) = STEP_REWARD
RLBase.UtilityStyle(::DDDCEnv) = GENERAL_SUM
RLBase.ChanceStyle(::DDDCEnv) = DETERMINISTIC

