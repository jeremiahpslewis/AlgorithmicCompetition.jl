using ReinforcementLearningCore
using ReinforcementLearningBase

const player_lookup = (; Symbol(1) => 1, Symbol(2) => 2)
const demand_lookup = (; :high => 1, :low => 2)
const player_to_index = (; Symbol(1) => 1, Symbol(2) => 2)
const demand_to_index = (; :high => 1, :low => 2)

"""
    AIAPCEnv(p::AIAPCHyperParameters)

    Build an environment to reproduce the results of the 2020  Calvano, Calzolari, Denicolò & Pastorello AER Paper

    Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020). Artificial Intelligence, Algorithmic Pricing, and Collusion. American Economic Review, 110(10), 3267–3297. https://doi.org/10.1257/aer.20190623
"""
struct AIAPCEnv <: AbstractEnv
    α::Float64                              # Learning parameter
    β::Float64                              # Exploration parameter
    δ::Float64                              # Discount factor
    max_iter::Int                           # Maximum number of iterations
    convergence_threshold::Int              # Convergence threshold

    n_players::Int                          # Number of players
    price_options::Vector{Float64}      # Price options
    price_index::Vector{Int8}           # Price indices

    competition_params_dict::Dict{Symbol,CompetitionParameters} # Competition parameters, true = high, false = low
    demand_mode::Symbol                      # Demand mode, :high or :low
    memory::Vector{CartesianIndex{2}}       # Memory vector (previous prices)
    state_space::Base.OneTo{Int16}          # State space
    state_space_lookup::Array{Int16,2}       # State space lookup table

    n_prices::Int                           # Number of price options
    n_state_space::Int64                    # Number of states

    convergence_vect::Vector{Bool}     # Convergence status for each player
    is_done::Vector{Bool}                # Episode is complete

    p_Bert_nash_equilibrium::Float64        # Nash equilibrium price (Betrand price)
    p_monop_opt::Float64                    # Monopoly optimal price

    action_space::Tuple                     # Action space
    profit_array::Array{Float64,3}          # Profit given price pair as coordinates

    function AIAPCEnv(p::AIAPCHyperParameters)
        price_options = Vector{Float64}(p.price_options)
        n_prices = length(p.price_options)
        price_index = Vector{Int8}(Int8.(1:n_prices))
        n_players = p.n_players
        n_state_space = n_prices^(p.memory_length * n_players)
        state_space = Base.OneTo(Int16(n_state_space))
        action_space = construct_AIAPC_action_space(price_index)
        profit_array = construct_AIAPC_profit_array(
            price_options,
            p.competition_params_dict,
            n_players;
            p.demand_mode,
        )
        state_space_lookup = construct_AIAPC_state_space_lookup(action_space, n_prices)

        @assert p.demand_mode ∈ (:high, :low)

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
            p.demand_mode,
            initialize_price_memory(price_index, p.n_players), # Memory, randomly initialized
            state_space,
            state_space_lookup,
            n_prices,
            n_state_space,
            Bool[false, false], # Convergence vector
            Vector{Bool}([false]), # Episode is done indicator
            p.p_Bert_nash_equilibrium,
            p.p_monop_opt,
            action_space,
            profit_array,
        )
    end
end

"""
    RLBase.act!(env::AIAPCEnv, price_tuple::Tuple{Int8,Int8})

Act in the environment by setting the memory to the given price tuple and setting `is_done` to `true`.
"""
function RLBase.act!(env::AIAPCEnv, price_tuple::CartesianIndex{2})
    # TODO: Fix support for longer memories
    env.memory[1] = price_tuple
    env.is_done[1] = true
end

RLBase.action_space(env::AIAPCEnv, ::Symbol) = env.price_index # Choice of price

RLBase.action_space(env::AIAPCEnv, ::SimultaneousPlayer) = env.action_space

RLBase.legal_action_space(env::AIAPCEnv, p) = is_terminated(env) ? () : action_space(env, p)

const legal_action_space_mask_object_AIAPC = [Int8.(1:15)...]

RLBase.legal_action_space_mask(env::AIAPCEnv, player::Symbol) =
    legal_action_space_mask_object_AIAPC

RLBase.action_space(env::AIAPCEnv) = action_space(env, SIMULTANEOUS_PLAYER)

"""
    RLBase.reward(env::AIAPCEnv)

Return the reward for the current state. If the episode is done, return the profit, else return `0, 0`.
"""
function RLBase.reward(env::AIAPCEnv)
    memory_index = env.memory[1]
    if env.is_done[1]
        return env.profit_array[memory_index, 1], env.profit_array[memory_index, 2]
    else
        return zero(Float64), zero(Float64)
    end
end

"""
    RLBase.reward(env::AIAPCEnv, p::Int)

Return the reward for the current state for player `p` as an integer. If the episode is done, return the profit, else return `0`.
"""
function RLBase.reward(env::AIAPCEnv, p::Int)
    profit_array = env.profit_array
    memory_index = env.memory[1]
    return _reward(profit_array, memory_index, p)
end

function _reward(profit::Array{Float64,3}, memory_index::CartesianIndex{2}, p::Int)

    return profit[memory_index, p]
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

    env.state_space_lookup[memory_index]
end

"""
    RLBase.is_terminated(env::AIAPCEnv)

Return whether the episode is done.
"""
RLBase.is_terminated(env::AIAPCEnv) = env.is_done[1]


function RLBase.reset!(env::AIAPCEnv)
    env.is_done[1] = false
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

function RLBase.plan!(
    explorer::Ex,
    learner::L,
    env::AIAPCEnv,
    player::Symbol,
) where {Ex<:AbstractExplorer,L<:AbstractLearner}
    legal_action_space_ = RLBase.legal_action_space_mask(env, player)
    return RLBase.plan!(
        explorer,
        RLCore.forward(learner, state(env, player)),
        legal_action_space_,
    )
end
