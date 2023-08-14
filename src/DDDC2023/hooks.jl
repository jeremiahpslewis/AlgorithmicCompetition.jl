struct DDDCRewardPerEpisodeLastN{F} <: AbstractHook where {F<:AbstractFloat}
    rewards::CircularBuffer{F}
    demand_state_high_vect::CircularBuffer{Bool}
    is_display_on_exit::Bool

    function DDDCRewardPerEpisodeLastN(; max_steps = 100)
        new{Float64}(CircularBuffer{Float64}(max_steps), CircularBuffer{Bool}(max_steps))
    end
end

Base.getindex(h::DDDCRewardPerEpisodeLastN{F}, inds...) where {F<:AbstractFloat} =
    getindex(h.rewards, inds...)

Base.push!(
    h::DDDCRewardPerEpisodeLastN{F},
    ::PostActStage,
    agent::P,
    env::E,
    player::Symbol,
) where {P<:AbstractPolicy,E<:AbstractEnv,F<:AbstractFloat} =
    h.rewards[end] += reward(env, player)


function Base.push!(
    hook::DDDCRewardPerEpisodeLastN{F},
    ::PreEpisodeStage,
    agent,
    env,
) where {F<:AbstractFloat}
    Base.push!(hook.rewards, 0.0)
    Base.push!(hook.demand_state_high_vect, env.is_high_demand_episode[1])
    return
end

function Base.push!(
    hook::DDDCRewardPerEpisodeLastN{F},
    stage::Union{PreEpisodeStage,PostEpisodeStage,PostExperimentStage},
    agent,
    env,
    player::Symbol,
) where {F<:AbstractFloat}
    Base.push!(hook, stage, agent, env)
    return
end

function Base.push!(
    hook::MultiAgentHook,
    stage::AbstractStage,
    policy::MultiAgentPolicy,
    env::DDDCEnv,
)
    @simd for p in (Symbol(1), Symbol(2))
        Base.push!(hook[p], stage, policy[p], env, p)
    end
end

function DDDCHook(env::AbstractEnv)
    MultiAgentHook(
        NamedTuple(
            p => ComposedHook(
                ConvergenceCheck(env.n_state_space, env.convergence_threshold),
                DDDCRewardPerEpisodeLastN(; max_steps = env.convergence_threshold + 100),
            ) for p in players(env)
        ),
    )
end
