# Code borrowed from RL.jl, only needed during RL.jl refactor, to be dropped once v0.11 is released.
include("stop_conditions.jl")
include("policy.jl")
include("named_policy.jl")
include("multi_agent.jl")
include("multi_agent_hook.jl")
include("linear_approximator.jl")
include("td_learner.jl")
include("reinforcement_learning_patch.jl")
