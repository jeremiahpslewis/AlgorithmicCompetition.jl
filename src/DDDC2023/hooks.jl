struct DDDCTotalRewardPerLastNEpisodes{F} <: AbstractHook where {F<:AbstractFloat}
    rewards::CircularBuffer{F}
    demand_state_high_vect::CircularBuffer{Bool}
    is_display_on_exit::Bool

    function DDDCTotalRewardPerLastNEpisodes(; max_steps = 100)
        new{Float64}(CircularBuffer{Float64}(max_steps), CircularBuffer{Bool}(max_steps))
    end
end

Base.getindex(h::DDDCTotalRewardPerLastNEpisodes{F}, inds...) where {F<:AbstractFloat} =
    getindex(h.rewards, inds...)

function Base.push!(
    h::DDDCTotalRewardPerLastNEpisodes{F},
    ::PostActStage,
    agent::P,
    env::E,
    player::Symbol,
) where {P<:AbstractPolicy,E<:AbstractEnv,F<:AbstractFloat}
    push!(h.rewards, reward(env, player))
    push!(h.demand_state_high_vect, env.memory.demand_state == :high)
end

function Base.push!(
    hook::DDDCTotalRewardPerLastNEpisodes{F},
    stage::Union{PreEpisodeStage,PostEpisodeStage,PostExperimentStage},
    agent,
    env,
    player::Symbol,
) where {F<:AbstractFloat}
    push!(hook, stage, agent, env)
    return
end

function Base.push!(
    hook::MultiAgentHook,
    stage::AbstractStage,
    policy::MultiAgentPolicy,
    env::DDDCEnv,
)
    @simd for p in (Symbol(1), Symbol(2))
        push!(hook[p], stage, policy[p], env, p)
    end
end

function DDDCHook(env::AbstractEnv)
    MultiAgentHook(
        NamedTuple(
            p => ComposedHook(
                ConvergenceCheck(env.n_state_space, env.convergence_threshold),
                DDDCTotalRewardPerLastNEpisodes(; max_steps = env.convergence_threshold + 100),
            ) for p in players(env)
        ),
    )
end
