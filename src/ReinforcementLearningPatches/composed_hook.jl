using ReinforcementLearningCore

"""
    ComposedHook(hooks::AbstractHook...)

Compose different hooks into a single hook.
"""
struct ComposedHook{T<:Tuple} <: AbstractHook
    hooks::T
    ComposedHook(hooks...) = new{typeof(hooks)}(hooks)
end

function update!(hook::ComposedHook, stage::AbstractStage, args...; kw...)
    for h in hook.hooks
        update!(h, stage, args...; kw...)
    end
    return
end

Base.getindex(hook::ComposedHook, inds...) = getindex(hook.hooks, inds...)
