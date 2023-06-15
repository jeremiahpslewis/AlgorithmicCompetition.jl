using ReinforcementLearningCore
using ReinforcementLearningBase
using StaticArrays
"""
    AIAPCEnv(p::AIAPCHyperParameters)

    Build an environment to reproduce the results of the 2020  Calvano, Calzolari, Denicolò & Pastorello AER Paper
    
    Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020). Artificial Intelligence, Algorithmic Pricing, and Collusion. American Economic Review, 110(10), 3267–3297. https://doi.org/10.1257/aer.20190623
"""
struct AIAPCEnv <: AbstractEnv
    α::Float64
    β::Float64
    δ::Float64
    n_players::Int
    memory_length::Int
    price_options::SVector{15,Float64}
    max_iter::Int
    convergence_threshold::Int
    n_prices::Int
    price_index::SVector{15,Int64}
    competition_solution::CompetitionSolution
    n_state_space::Int64
    state_space::Base.OneTo{Int64}
    state_space_lookup::Matrix{Int64}
    memory::MVector{2,Int}
    convergence_dict::Dict{Symbol,Bool}
    is_done::MVector{1,Bool}
    p_Bert_nash_equilibrium::Float64
    p_monop_opt::Float64
    action_space::Tuple
    profit_array::Array{Float64}

    function AIAPCEnv(p::AIAPCHyperParameters)
        price_options = SVector{15,Float64}(p.price_options)
        n_prices = length(p.price_options)
        price_index = SVector{15,Int64}(1:n_prices)
        n_players = p.n_players
        n_state_space = n_prices^(p.memory_length * n_players)
        state_space = Base.OneTo(Int64(n_state_space))
        action_space = Tuple((i, j) for i in price_index for j in price_index)

        profit_array =
            construct_profit_array(price_options, p.competition_solution.params, n_players)
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
            p.competition_solution,
            n_state_space,
            state_space,
            state_space_lookup,
            MVector{2,Int}(rand(price_index, p.memory_length, p.n_players)), # Memory, randomly initialized
            Dict(Symbol(1) => false, Symbol(2) => false), # Convergence counter
            MVector{1,Bool}([false]), # Is done
            p.p_Bert_nash_equilibrium,
            p.p_monop_opt,
            action_space,
            profit_array,
        )
    end
end

# TODO: add back type signature ::Tuple{Int64,Int64}
function RLBase.act!(env::AIAPCEnv, price_tuple::Tuple{Int64,Int64})
    # TODO: Fix support for longer memories
    env.memory .= price_tuple
    env.is_done[1] = true
end

function construct_state_space_lookup(action_space, n_prices)
    @assert length(action_space) == n_prices^2
    state_space_lookup = reshape(1:length(action_space), n_prices, n_prices)
    return state_space_lookup
end

function construct_profit_array(
    price_options::SVector{15,Float64},
    params::CompetitionParameters,
    n_players::Int,
)
    n_prices = length(price_options)
    # TODO: Carve out into separate function:
    profit_array = zeros(Float64, n_prices, n_prices, n_players)
    for k = 1:n_players
        for i = 1:n_prices
            for j = 1:n_prices
                # TODO: Check that player assignment is correct here (should be...?)
                profit_array[i, j, k] = π(price_options[i], price_options[j], params)[k]
            end
        end
    end

    return profit_array
end

RLBase.action_space(env::AIAPCEnv, ::Symbol) = env.price_index # Choice of price

RLBase.action_space(env::AIAPCEnv, ::SimultaneousPlayer) = env.action_space

RLBase.legal_action_space(env::AIAPCEnv, p) = is_terminated(env) ? () : action_space(env, p)

RLBase.legal_action_space_mask(env::AIAPCEnv, player::Symbol) = SA[1:15...]

RLBase.action_space(env::AIAPCEnv) = action_space(env, SIMULTANEOUS_PLAYER)

function RLBase.reward(env::AIAPCEnv)
    env.is_done[1] ? (@view env.profit_array[env.memory[1], env.memory[2], :]) : SA[0, 0]
end

function RLBase.reward(env::AIAPCEnv, p::Int)
    (@view env.profit_array[env.memory[1], env.memory[2], p])[1]
end


const player_lookup = (; Symbol(1) => 1, Symbol(2) => 2)

RLBase.reward(env::AIAPCEnv, p::Symbol) = reward(env, player_lookup[p])

RLBase.state_space(env::AIAPCEnv, ::Observation, p) = env.state_space

function RLBase.state(env::AIAPCEnv, ::Observation, p)
    env.state_space_lookup[env.memory[1], env.memory[2]]
end

RLBase.is_terminated(env::AIAPCEnv) = env.is_done[1]
# TODO: Expand reset function to other params
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
