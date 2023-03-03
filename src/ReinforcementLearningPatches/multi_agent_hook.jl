using ReinforcementLearningCore

"""
    MultiAgentHook(player=>hook...)
"""
struct MultiAgentHook <: AbstractHook
    hooks::Dict{Any,Any}
end

MultiAgentHook(player_hook_pair::Pair...) = MultiAgentHook(Dict(player_hook_pair...))

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
end
