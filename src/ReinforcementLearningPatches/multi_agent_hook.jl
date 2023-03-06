using ReinforcementLearningCore

"""
    MultiAgentHook(player=>hook...)
"""
struct MultiAgentHook
    hooks::NamedTuple{AbstractHook}
end

MultiAgentHook(player_hook_pair::Pair...) = MultiAgentHook(NamedTuple{AbstractHook}(player_hook_pair))

Base.getindex(h::MultiAgentHook, p) = getindex(h.hooks, p)

function (hook::MultiAgentHook{K})(
    s::AbstractStage,
    m::MultiAgentManager{K},
    env::AbstractEnv,
    args...,
) where {K}
    for (agent, policy) in m.agents
        update!(hook.hooks[agent], s, policy, env, args...)
    end

    return
end
