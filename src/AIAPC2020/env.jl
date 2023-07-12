using ReinforcementLearningCore
using ReinforcementLearningBase
using StaticArrays

const player_lookup = (; Symbol(1) => 1, Symbol(2) => 2)
const demand_lookup = (; :high => 1, :low => 2)
const player_to_index = (; Symbol(1) => 1, Symbol(2) => 2)
const demand_to_index = (; :high => 1, :low => 2)

"""
    AIAPCEnv(p::AIAPCHyperParameters)

    Build an environment to reproduce the results of the 2020  Calvano, Calzolari, Denicolò & Pastorello AER Paper

    Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020). Artificial Intelligence, Algorithmic Pricing, and Collusion. American Economic Review, 110(10), 3267–3297. https://doi.org/10.1257/aer.20190623
"""
struct AIAPCEnv{N,M} <: AbstractEnv # N is profit_array dimension, M is state_space_lookup dimension, N = M + 1
    α::Float64                              # Learning parameter
    β::Float64                              # Exploration parameter
    δ::Float64                              # Discount factor
    max_iter::Int                           # Maximum number of iterations
    convergence_threshold::Int              # Convergence threshold

    n_players::Int                          # Number of players
    price_options::SVector{15,Float64}      # Price options
    price_index::SVector{15,Int8}           # Price indices

    competition_params_dict::Dict{Symbol, CompetitionParameters} # Competition parameters, true = high, false = low
    activate_extension::Bool                # Whether to activate the Data/Demand/Digital extension
    demand_mode::Symbol                      # Demand mode, :high, :low, or :random
    memory::Vector{CartesianIndex{M}}       # Memory vector (previous prices)
    is_high_demand_signals::Vector{Bool}    # [true, false] if demand signal is high for player one and low for player two for a given episode
    is_high_demand_episode::Vector{Bool}    # [true] if demand is high for a given episode
    state_space::Base.OneTo{Int16}          # State space
    state_space_lookup::Matrix{Int16}       # State space lookup table

    n_prices::Int                           # Number of price options
    n_state_space::Int64                    # Number of states

    convergence_dict::Dict{Symbol,Bool}     # Convergence status for each player
    is_done::MVector{1,Bool}                # Episode is complete

    p_Bert_nash_equilibrium::Float64        # Nash equilibrium price (Betrand price)
    p_monop_opt::Float64                    # Monopoly optimal price

    action_space::Tuple                     # Action space
    profit_array::Array{Float64,N}          # Profit given price pair as coordinates

    data_demand_digital_params::DataDemandDigitalParams # Parameters for Data/Demand/Digital AIAPC extension

    function AIAPCEnv(p::AIAPCHyperParameters;
        data_demand_digital_params::DataDemandDigitalParams = DataDemandDigitalParams())
        price_options = SVector{15,Float64}(p.price_options)
        n_prices = length(p.price_options)
        price_index = SVector{15,Int8}(Int8.(1:n_prices))
        n_players = p.n_players
        n_state_space = n_prices^(p.memory_length * n_players)
        state_space = Base.OneTo(Int16(n_state_space))
        action_space = construct_action_space(price_index, p.activate_extension)
        profit_array =
            construct_profit_array(price_options, p.competition_params_dict, n_players; p.activate_extension, data_demand_digital_params.demand_mode)
        state_space_lookup = construct_state_space_lookup(action_space, n_prices, p.activate_extension)
        is_high_demand_episode = rand(Bool, 1)

        @assert data_demand_digital_params.demand_mode ∈ (:high, :low, :random)
        @assert data_demand_digital_params.demand_mode == :random || p.activate_extension == false

        if p.activate_extension
            n = Int64(4)
            m = Int64(3)
        else
            n = Int64(3)
            m = Int64(2)
        end

        new{n,m}(
            p.α,
            p.β,
            p.δ,
            p.max_iter,
            p.convergence_threshold,
            n_players,
            p.price_options,
            price_index,
            p.competition_params_dict,
            p.activate_extension,
            data_demand_digital_params.demand_mode,
            initialize_price_memory(price_index, p.n_players), # Memory, randomly initialized
            get_demand_signals(data_demand_digital_params, is_high_demand_episode[1]),
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
            data_demand_digital_params,
        )
    end
end

"""
    RLBase.act!(env::AIAPCEnv, price_tuple::Tuple{Int8,Int8})

Act in the environment by setting the memory to the given price tuple and setting `is_done` to `true`.
"""
function RLBase.act!(env::AIAPCEnv{N,M}, price_tuple::CartesianIndex{M}) where {N,M}
    # TODO: Fix support for longer memories
    env.memory[1] = price_tuple
    env.is_done[1] = true
end

"""
    construct_state_space_lookup(action_space, n_prices)

Construct a lookup table from action space to the state space.
"""
function construct_state_space_lookup(action_space, n_prices, activate_extension = false)
    if activate_extension
        @assert length(action_space) == n_prices^2 * 2
        state_space_lookup = reshape(Int16.(1:length(action_space)), n_prices, n_prices, 2, 2)
        return state_space_lookup
    else
        @assert length(action_space) == n_prices^2
        state_space_lookup = reshape(Int16.(1:length(action_space)), n_prices, n_prices)
        return state_space_lookup
    end
end


"""
    construct_profit_array(price_options, params, n_players; activate_extension, demand_mode)

Construct a 3-dimensional array which holds the profit for each player given a price pair.
The first dimension is player 1's action, the second dimension is player 2's action, and
the third dimension is the player index for their profit.
"""
function construct_profit_array(
    price_options::SVector{15,Float64},
    competition_params_dict::Dict{Symbol,CompetitionParameters},
    n_players::Int;
    activate_extension = false,
    demand_mode = :high,
)
    n_prices = length(price_options)

    if activate_extension
        profit_array = zeros(Float64, n_prices, n_prices, n_players, 2)
        for l = [:high, :low]
            for k = 1:n_players
                for i = 1:n_prices
                    for j = 1:n_prices
                        profit_array[i, j, k, demand_lookup[l]] = π(price_options[i], price_options[j], competition_params_dict[l])[k]
                    end
                end
            end
        end
    else
        params_ = competition_params_dict[demand_mode]

        profit_array = zeros(Float64, n_prices, n_prices, n_players)
        for k = 1:n_players
            for i = 1:n_prices
                for j = 1:n_prices
                    profit_array[i, j, k] = π(price_options[i], price_options[j], params_)[k]
                end
            end
        end
    end

    return profit_array
end

RLBase.action_space(env::AIAPCEnv, ::Symbol) = env.price_index # Choice of price

RLBase.action_space(env::AIAPCEnv, ::SimultaneousPlayer) = env.action_space

RLBase.legal_action_space(env::AIAPCEnv, p) = is_terminated(env) ? () : action_space(env, p)

const legal_action_space_mask_object = [Int8.(1:15)...]

RLBase.legal_action_space_mask(env::AIAPCEnv, player::Symbol) =
    legal_action_space_mask_object

RLBase.action_space(env::AIAPCEnv) = action_space(env, SIMULTANEOUS_PLAYER)

const zero_tuple = Tuple{Float64,Float64}([0, 0])

"""
    RLBase.reward(env::AIAPCEnv)

Return the reward for the current state. If the episode is done, return the profit, else return `(0, 0)`.
"""
function RLBase.reward(env::AIAPCEnv)
    if env.activate_extension
        memory_index = env.memory[1]
        env.is_done[1] ? Tuple{Float64,Float64}(env.profit_array[memory_index, :, demand_to_index[env.demand_signal]]) : zero_tuple
    else
        memory_index = env.memory[1]
        env.is_done[1] ? Tuple{Float64,Float64}(env.profit_array[memory_index, :]) : zero_tuple
    end
end

"""
    RLBase.reward(env::AIAPCEnv, p::Int)

Return the reward for the current state for player `p` as an integer. If the episode is done, return the profit, else return `0`.
"""
function RLBase.reward(env::AIAPCEnv{N,M}, p::Int)::Float64 where {N,M}
    profit_array = env.profit_array
    memory_index = env.memory[1]
    return _reward(
        profit_array,
        memory_index,
        env.activate_extension,
        env.is_high_demand_episode[1],
        p
        )
end

function _reward(profit::Array{Float64,M},
    memory_index::CartesianIndex{N},
    activate_extension::Bool,
    is_high_demand_episode::Bool,
    p::Int)::Float64 where {N,M}
    if is_high_demand_episode
        demand_index_ = 2
    else
        demand_index_ = 1
    end

    if activate_extension
        return profit[memory_index, p, demand_index_]
    else
        return profit[memory_index, p]
    end
end

"""
    RLBase.reward(env::AIAPCEnv, p::Int)

Return the reward for the current state for player `p` as a symbol. If the episode is done, return the profit, else return `0`.
"""
RLBase.reward(env::AIAPCEnv, p::Symbol) = reward(env, player_lookup[p])

RLBase.state_space(env::AIAPCEnv, ::Observation, p) = env.state_space

# State without player spec is a noop
RLBase.state(env::AIAPCEnv) = nothing

"""
    RLBase.state(env::AIAPCEnv, player::Symbol)

Return the current state as an integer, mapped from the environment memory.
"""
function RLBase.state(env::AIAPCEnv, p::Symbol)
    memory_index = env.memory[1]
    if env.activate_extension
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
    else
        env.state_space_lookup[memory_index]
    end
end

"""
    RLBase.is_terminated(env::AIAPCEnv)

Return whether the episode is done.
"""
RLBase.is_terminated(env::AIAPCEnv) = env.is_done[1]


function RLBase.reset!(env::AIAPCEnv)
    env.is_done[1] = false

    # For data / demand / digital extension...
    if env.activate_extension
        # Determine whether next episode is a high demand episode
        is_high_demand_episode = get_demand_level(env.data_demand_digital_params)

        # Update demand signals
        env.prev_is_high_demand_signals .= env.is_high_demand_signals
        env.is_high_demand_signals .= get_demand_signals(env.data_demand_digital_params, is_high_demand_episode)

        # Update demand level
        env.is_high_demand_episode[1] = is_high_demand_episode


    end
end

RLBase.players(::AIAPCEnv) = (Symbol(1), Symbol(2))
RLBase.current_player(::AIAPCEnv) = SIMULTANEOUS_PLAYER
RLBase.NumAgentStyle(::AIAPCEnv) = MultiAgent(2)
RLBase.DynamicStyle(::AIAPCEnv) = SIMULTANEOUS
RLBase.ActionStyle(::AIAPCEnv) = MINIMAL_ACTION_SET
RLBase.InformationStyle(::AIAPCEnv) = IMPERFECT_INFORMATION
RLBase.StateStyle(::AIAPCEnv) = Observation{Int64}()
RLBase.RewardStyle(::AIAPCEnv) = STEP_REWARD
RLBase.UtilityStyle(::AIAPCEnv) = GENERAL_SUM
RLBase.ChanceStyle(::AIAPCEnv) = DETERMINISTIC

function RLBase.plan!(explorer::Ex, learner::L, env::AIAPCEnv, player::Symbol) where {Ex<:AbstractExplorer,L<:AbstractLearner,E<:AbstractEnv}
    legal_action_space_ = RLBase.legal_action_space_mask(env, player)
    return RLBase.plan!(explorer, RLCore.forward(learner, state(env, player)), legal_action_space_)
end
