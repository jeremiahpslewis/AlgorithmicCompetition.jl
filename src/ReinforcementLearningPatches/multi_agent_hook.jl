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
        push!(agent_names, Symbol(pair.first))
        push!(agent_hooks, pair.second)
    end
    MultiAgentHook(agent_names, agent_hooks)
end

Base.getindex(h::MultiAgentHook, p) = getindex(h.hooks, p)

function (hook::MultiAgentHook)(
    s::AbstractStage,
    m::MultiAgentManager,
    env::AbstractEnv,
    args...,
)
    for i in 1:length(m.agent_policies)
        update!(hook.agent_hooks[i], s, m.agent_policies[i], env, args...)
    end

    return
end
