using ReinforcementLearning

Base.@kwdef mutable struct CalvanoEnv <: AbstractEnv
    params::CalvanoParams
    memory::Array{Int64,2}
    is_converged::Base.AbstractVecOrTuple{Bool}
    reward::Tuple{Float64,Float64} = (0.0, 0.0) # Placeholder
    is_done::Bool = false

    function CalvanoEnv(params::CalvanoParams)
        # Special case starting conditions with 'missing' in lookbacks, think about best way of handling this...
        # TODO: Think about how initial memory should be assigned
        new(params,
            fill(1, params.memory_length, params.n_players),
            ntuple((i) -> false, params.n_players)
        )
    end
end



function (env::CalvanoEnv)((p_1, p_2))
    # Convert from price indices to price level, compute profit
    env.reward = π_fun(env.params.price_options[p_1], env.params.price_options[p_2]) |> Tuple

    env.memory = circshift(env.memory, -1)
    env.memory[end, :] = [p_1, p_2]

    env.is_done = true
end

function map_memory_to_state(v, n_prices)
    v = vec(v)
    sum((v .- 1) .* n_prices .^ ((1:length(v)) .- 1)) + 1
end

RLBase.action_space(env::CalvanoEnv, ::Int) = env.params.price_index # Choice of price

RLBase.action_space(::CalvanoEnv, ::SimultaneousPlayer) =
    Tuple((i, j) for i in env.params.price_index for j in env.params.price_index)

RLBase.legal_action_space(env::CalvanoEnv, p) =
    is_terminated(env) ? () : action_space(env, p)

RLBase.action_space(env::CalvanoEnv) = action_space(env, SIMULTANEOUS_PLAYER)

RLBase.reward(env::CalvanoEnv) = env.is_done ? env.reward : (0, 0)
RLBase.reward(env::CalvanoEnv, p::Int) = reward(env)[p]

RLBase.state_space(::CalvanoEnv, ::Observation, p) = Base.OneTo(n_state_space)

function RLBase.state(env::CalvanoEnv, ::Observation, p)
    map_memory_to_state(env.memory, env.params.n_prices)
end

RLBase.is_terminated(env::CalvanoEnv) = env.is_done
# TODO: Expand reset function to other params
RLBase.reset!(env::CalvanoEnv) = env.is_done = false
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
