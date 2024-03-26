using CircularArrayBuffers

struct DDDCTotalRewardPerLastNEpisodes{B} <: AbstractHook where {B<:CircularArrayBuffer}
    rewards::B
    demand_state_high_vect::CircularVectorBuffer{Bool}

    function DDDCTotalRewardPerLastNEpisodes(; max_steps = 100)
        new{Float64}(CircularVectorBuffer{Float64}(max_steps), CircularVectorBuffer{Bool}(max_steps))
    end
end

Base.getindex(h::DDDCTotalRewardPerLastNEpisodes{F}, inds...) where {F<:AbstractFloat} =
    getindex(h.rewards, inds...)

function Base.push!(
    h::DDDCTotalRewardPerLastNEpisodes{F},
    ::PostActStage,
    agent::P,
    env::E,
    player::Player,
) where {P<:AbstractPolicy,E<:AbstractEnv,F<:AbstractFloat}
    push!(h.rewards, reward(env, player))
    push!(h.demand_state_high_vect, env.memory.demand_state == :high)
end

function Base.push!(
    hook::DDDCTotalRewardPerLastNEpisodes{F},
    stage::Union{PreEpisodeStage,PostEpisodeStage,PostExperimentStage},
    agent::P,
    env::E,
    player::Player,
) where {P<:AbstractPolicy,E<:AbstractEnv,F<:AbstractFloat}
    push!(hook, stage, agent, env)
    return
end

function Base.push!(
    hook::MultiAgentHook,
    stage::AbstractStage,
    policy::MultiAgentPolicy,
    env::DDDCEnv,
)
    @simd for p in (Player(1), Player(2))
        push!(hook[p], stage, policy[p], env, p)
    end
end

function DDDCHook(env::AbstractEnv)
    MultiAgentHook(
        PlayerTuple(
            p => ComposedHook(
                ConvergenceCheck(env.n_state_space, env.convergence_threshold),
                DDDCTotalRewardPerLastNEpisodes(; max_steps = env.convergence_threshold + 100),
            ) for p in players(env)
        ),
    )
end
