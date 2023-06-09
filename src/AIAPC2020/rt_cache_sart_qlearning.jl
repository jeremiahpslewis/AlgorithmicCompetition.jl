import Base.push!
using ReinforcementLearningTrajectories
using ReinforcementLearningCore

mutable struct ART{A,R,T}
    action::Union{A,Nothing}
    reward::Union{R,Nothing}
    terminal::Union{T,Nothing}

    function ART()
        new{Any, Any, Any}(nothing, nothing, nothing)
    end

    function ART{A,R,T}() where {A,R,T}
        new{A,R,T}(nothing, nothing, nothing)
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
    push!(agent.cache, action)
    action
end

function Base.push!(t::Trajectory, art::ART{R,T}, state::S) where {S,R,T}
    traces = t.container.traces
    push!(traces[1].trace, state) # Push state

    reward = art.reward
    terminal = art.terminal
    action = art.action
    if !isnothing(reward) && !isnothing(terminal) && !isnothing(action)
        push!(traces[2], action)
        push!(traces[3], reward)
        push!(traces[4], terminal)
    end
    ReinforcementLearningTrajectories.on_insert!(t) # let trajectory know it has been updated (required for batch sampling to work)
end

function Base.push!(cache::ART{A,R,T}, action::A) where {A,R,T}
    cache.action = action
end

function Base.push!(cache::ART{A,R,T}, reward::R, terminal::T) where {A,R,T}
    cache.reward = reward
    cache.terminal = terminal
end

function RLBase.reset!(cache::ART)
    cache.reward = nothing
    cache.terminal = nothing
end