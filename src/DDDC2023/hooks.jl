using CircularArrayBuffers

struct DDDCTotalRewardPerLastNEpisodes <: AbstractHook
    rewards::CircularVectorBuffer{Float64}
    demand_state_high_vect::CircularVectorBuffer{Bool}

    function DDDCTotalRewardPerLastNEpisodes(; max_steps = 100)
        new(CircularVectorBuffer{Float64}(max_steps), CircularVectorBuffer{Bool}(max_steps))
    end
end

function Base.push!(h::DDDCTotalRewardPerLastNEpisodes,
    env::E,
    memory::DDDCMemory,
    player::Player
) where {E<:AbstractEnv}
    push!(h.rewards, reward(env, player))
    push!(h.demand_state_high_vect, memory.demand_state == :high)
end

function Base.push!(
    h::DDDCTotalRewardPerLastNEpisodes,
    ::PostActStage,
    agent::P,
    env::E,
    player::Player,
) where {P<:AbstractPolicy,E<:AbstractEnv}
    push!(h, env, env.memory, player)
end

function Base.push!(
    hook::DDDCTotalRewardPerLastNEpisodes,
    stage::Union{PreEpisodeStage,PostEpisodeStage,PostExperimentStage},
    agent::P,
    env::E,
    player::Player,
) where {P<:AbstractPolicy,E<:AbstractEnv}
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
