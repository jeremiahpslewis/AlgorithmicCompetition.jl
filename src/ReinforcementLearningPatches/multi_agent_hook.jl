using ReinforcementLearningCore

"""
    MultiAgentHook(player=>hook...)
"""
struct MultiAgentHook
    agent_names::Vector{Symbol}
    agent_hooks::Vector{AbstractHook}
end

function MultiAgentHook(player_hook_pair::Pair...)
    agent_names = Symbol[]
    agent_hooks = AbstractHook[]

    for pair in player_hook_pair
        push!(agent_names, p.first)
        push!(agent_hooks, p.second)
    end
    MultiAgentHook(agent_names, agent_hooks)
end

Base.getindex(h::MultiAgentHook, p) = getindex(h.hooks, p)

function (hook::MultiAgentHook)(
    s::AbstractStage,
    m::MultiAgentManager,
    env::AbstractEnv,
    args...,
) where {K}
    for (agent, policy) in m.agents
        update!(hook.agent_hooks[agent], s, policy, env, args...)
    end

    return
end
