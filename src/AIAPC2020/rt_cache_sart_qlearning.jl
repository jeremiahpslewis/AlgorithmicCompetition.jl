import Base.push!
using ReinforcementLearningTrajectories
using ReinforcementLearningCore

mutable struct RT{R,T}
    reward::Union{R,Nothing}
    terminal::Union{T,Nothing}

    function RT()
        new{Any, Any}(nothing, nothing)
    end

    function RT{R,T}() where {R,T}
        new{R,T}(nothing, nothing)
    end
end

function Base.push!(multiagent::MultiAgentPolicy, ::PreActStage, env::AIAPCEnv)
    for player in players(env)
        agent = multiagent[player]
        push!(agent.trajectory, agent.cache, state(env, player))
    end
end

function RLBase.plan!(agent::Agent{P,T,C}, env::AIAPCEnv, p::Symbol) where {P,T,C}
    action = RLBase.plan!(agent.policy, env, p)
    traces = agent.trajectory.container.traces
    push!(traces[2].trace, action)
    action
end

function Base.push!(t::Trajectory, rt::RT{R,T}, state::S) where {S,R,T}
    traces = t.container.traces
    push!(traces[1].trace, state) # Push state
    if !isnothing(rt.reward) && !isnothing(rt.terminal)
        push!(traces[3], rt.reward)
        push!(traces[4], rt.terminal)
    end
end

function Base.push!(cache::RT{R,T}, reward::R, terminal::T) where {R,T}
    cache.reward = reward
    cache.terminal = terminal
end

function RLBase.reset!(cache::RT)
    cache.reward = nothing
    cache.terminal = nothing
end
