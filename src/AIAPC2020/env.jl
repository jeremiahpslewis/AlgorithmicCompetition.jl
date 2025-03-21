using ReinforcementLearning

"""
    AIAPCEnv(p::AIAPCHyperParameters)

    Build an environment to reproduce the results of the 2020  Calvano, Calzolari, Denicolò & Pastorello AER Paper

    Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020). Artificial Intelligence, Algorithmic Pricing, and Collusion. American Economic Review, 110(10), 3267–3297. https://doi.org/10.1257/aer.20190623
"""
struct AIAPCEnv <: AbstractEnv
    α::Float64                               # Learning parameter
    β::Float64                               # Exploration parameter
    δ::Float64                               # Discount factor
    max_iter::Int                            # Maximum number of iterations
    convergence_threshold::Int               # Convergence threshold

    n_players::Int                           # Number of players
    price_options::Vector{Float64}           # Price options
    price_index::Vector{Int64}                # Price indices

    competition_params_dict::Dict{Symbol,CompetitionParameters} # Competition parameters, true = high, false = low
    demand_mode::Symbol                      # Demand mode, :high or :low
    memory::Vector{CartesianIndex{2}}        # Memory vector (previous prices)
    state_space::Base.OneTo{Int64}           # State space
    state_space_lookup::Array{Int64,2}       # State space lookup table

    n_prices::Int                            # Number of price options
    n_state_space::Int64                     # Number of states

    convergence_vect::Vector{Bool}           # Convergence status for each player
    is_done::Vector{Bool}                    # Episode is complete

    p_Bert_nash_equilibrium::Float64         # Nash equilibrium price (Betrand price)
    p_monop_opt::Float64                     # Monopoly optimal price

    action_space::Tuple                      # Action space
    profit_array::Array{Float64,3}           # Profit given price pair as coordinates

    reward::Vector{Float64}                  # Reward vector

    function AIAPCEnv(p::AIAPCHyperParameters)
        price_options = Vector{Float64}(p.price_options)
        n_prices = length(p.price_options)
        price_index = Vector{Int64}(Int64.(1:n_prices))
        n_players = p.n_players
        n_state_space = n_prices^(p.memory_length * n_players)
        state_space = Base.OneTo(Int64(n_state_space))
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
            CartesianIndex{2}[initialize_price_memory(price_index, p.n_players)], # Memory, randomly initialized
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
            Float64[0.0, 0.0], # Reward vector
        )
    end
end

"""
    RLBase.act!(env::AIAPCEnv, price_tuple::Tuple{Int64,Int64})

Act in the environment by setting the memory to the given price tuple and setting `is_done` to `true`.
"""
function RLBase.act!(env::AIAPCEnv, price_tuple::CartesianIndex{2})
    # TODO: Fix support for longer memories
    memory_index = env.memory[1]
    env.reward .= env.profit_array[memory_index, :]

    env.memory[1] = price_tuple
    env.is_done[1] = true
end

RLBase.action_space(env::AIAPCEnv, ::Player) = env.price_index # Choice of price

RLBase.action_space(env::AIAPCEnv, ::SimultaneousPlayer) = env.action_space

RLBase.legal_action_space(env::AIAPCEnv, p) = is_terminated(env) ? () : action_space(env, p)

const legal_action_space_mask_object_AIAPC = fill(true, 15)

RLBase.legal_action_space_mask(env::AIAPCEnv, player::Player) =
    legal_action_space_mask_object_AIAPC

RLBase.action_space(env::AIAPCEnv) = action_space(env, SIMULTANEOUS_PLAYER)

"""
    RLBase.reward(env::AIAPCEnv)

Return the reward for the current state. If the episode is done, return the profit, else return `0, 0`.
"""
function RLBase.reward(env::AIAPCEnv)
    env.is_done[1] ? env.reward : (zero(Float64), zero(Float64))
end

"""
    RLBase.reward(env::AIAPCEnv, p::Int)

Return the reward for the current state for player `p` as an integer. If the episode is done, return the profit, else return `0`.
"""
function RLBase.reward(env::AIAPCEnv, p::Int)
    return env.reward[p]
end

"""
    RLBase.reward(env::AIAPCEnv, player::Player)

Return the reward for the current state for `player`. If the episode is done, return the profit, else return `0`.
"""
RLBase.reward(env::AIAPCEnv, p::Player) = reward(env, player_to_index[p])

RLBase.state_space(env::AIAPCEnv, ::Observation, p) = env.state_space

# State without player spec is a noop
RLBase.state(env::AIAPCEnv) = nothing

"""
    RLBase.state(env::AIAPCEnv, player::Player)

Return the current state as an integer, mapped from the environment memory.
"""
function RLBase.state(env::AIAPCEnv, player::Player)
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

const players_ = (Player(1), Player(2))
RLBase.players(::AIAPCEnv) = players_
RLBase.current_player(::AIAPCEnv) = SIMULTANEOUS_PLAYER
RLBase.NumAgentStyle(::AIAPCEnv) = MultiAgent(2)
RLBase.DynamicStyle(::AIAPCEnv) = SIMULTANEOUS
RLBase.ActionStyle(::AIAPCEnv) = MINIMAL_ACTION_SET
RLBase.InformationStyle(::AIAPCEnv) = IMPERFECT_INFORMATION
RLBase.StateStyle(::AIAPCEnv) = Observation{Int64}()
RLBase.RewardStyle(::AIAPCEnv) = STEP_REWARD
RLBase.UtilityStyle(::AIAPCEnv) = GENERAL_SUM
RLBase.ChanceStyle(::AIAPCEnv) = DETERMINISTIC


# Need special handling of episodes and experiments for the AIAPC and DDDC environments: an episode is a single price setting interaction, and an experiment is a sequence of episodes, but the environment state is not reset between episodes. As a result, the state is initialized once, in the PreExperimentStage and PreEpisodeStage becomes a no-op.
Base.push!(agent::Agent, ::PreEpisodeStage, env::AIAPCEnv, player::Player) = nothing

function Base.push!(agent::Agent, ::PreExperimentStage, env::AIAPCEnv, player::Player)
    push!(agent.trajectory, (state = state(env, player),))
end
