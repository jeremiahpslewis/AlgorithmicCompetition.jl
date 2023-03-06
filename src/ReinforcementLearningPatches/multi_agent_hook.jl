using ReinforcementLearningCore

"""
    MultiAgentHook(player=>hook...)
"""
struct MultiAgentHook{K}
    hooks::Dict{K,AbstractHook}
end

MultiAgentHook{K}(player_hook_pair::Pair...) where {K} = MultiAgentHook{K}(Dict{K, AbstractHook}(player_hook_pair...))

Base.getindex(h::MultiAgentHook, p) = getindex(h.hooks, p)

function (hook::MultiAgentHook)(
    s::AbstractStage,
    m::MultiAgentManager,
    env::AbstractEnv,
    args...,
)
    for (p, h) in zip(values(m.agents), values(hook.hooks))
        h(s, p, env, args...)
    end

    return
end
