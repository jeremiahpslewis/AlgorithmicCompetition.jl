export MultiAgentManager, NO_OP, NoOp

using ReinforcementLearningBase
using ReinforcementLearningCore: AbstractPolicy, AbstractStage, AbstractEnv, PreActStage, DynamicStyle, Sequential, Simultaneous, current_player


"Represent no-operation if it's not the agent's turn."
struct NoOp end

const NO_OP = NoOp()

struct MultiAgentManager <: AbstractPolicy
    agent_policies::Vector{NamedPolicy}
end

Base.getindex(A::MultiAgentManager, x) = getindex(A.agent_policies, x)

"""
    MultiAgentManager(player => policy...)

This is the simplest form of multiagent system. At each step they observe the
environment from their own perspective and get updated independently. For
environments of `SEQUENTIAL` style, agents which are not the current player will
observe a dummy action of [`NO_OP`](@ref) in the `PreActStage`. For environments
of `SIMULTANEOUS` style, please wrap it with [`SequentialEnv`](@ref) first.
"""

RLBase.prob(A::MultiAgentManager, env::AbstractEnv, args...) = prob(A[current_player(env)].policy, env, args...)

(A::MultiAgentManager)(env::AbstractEnv) = A(env, DynamicStyle(env))

(A::MultiAgentManager)(env::AbstractEnv, ::Sequential) = A[current_player(env)](env)

function (A::MultiAgentManager)(env::AbstractEnv, ::Simultaneous)
    @error "MultiAgentManager doesn't support simultaneous environments. Please consider applying `SequentialEnv` wrapper to environment first."
end

function (A::MultiAgentManager)(stage::AbstractStage, env::AbstractEnv)
    for agent in A
        agent(stage, env)
    end
end

function RLBase.optimise!(A::MultiAgentManager)
    for agent in A
        RLBase.optimise!(agent)
    end
end

function (A::MultiAgentManager)(stage::PreActStage, env::AbstractEnv, action)
    A(stage, env, DynamicStyle(env), action)
end

function (A::MultiAgentManager)(stage::PreActStage, env::AbstractEnv, ::Sequential, action)
    p = current_player(env)
    for agent in A
        if agent.name == Symbol(player)
            agent(stage, env, action)
        else
            agent(stage, env, NO_OP)
        end
    end
end

function (A::MultiAgentManager)(
    stage::PreActStage,
    env::AbstractEnv,
    ::Simultaneous,
    actions,
)
    for (agent, action) in zip(A, actions)
        agent(stage, env, action)
    end
end
