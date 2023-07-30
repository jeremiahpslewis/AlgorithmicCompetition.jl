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
