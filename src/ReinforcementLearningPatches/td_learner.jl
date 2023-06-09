export TDLearner

using LinearAlgebra: dot
using Distributions: pdf
using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningTrajectories: Trajectory


function TDLearner(; approximator::Ap, γ::F = 1.0, method::Symbol, n::I = 0) where {Ap,F,I}
    if method != :SARS
        @error "unsupported method"
    else
        TDLearnerSARS{Ap,F,I}(approximator, γ, method, n)
    end
end

struct TDLearnerSARS{Ap,F<:AbstractFloat,I<:Integer} <: AbstractLearner
    approximator::Ap
    γ::F
    method::Symbol
    n::I
end

RLCore.forward(L::TDLearnerSARS{Ap,F,I}, env::E) where {Ap,F,I,E<:AbstractEnv} =
    RLCore.forward(L.approximator, state(env))
RLCore.forward(L::TDLearnerSARS{Ap,F,I}, s::I1) where {Ap,F,I<:Integer,I1<:Integer} =
    RLCore.forward(L.approximator, s)
RLCore.forward(
    L::TDLearnerSARS{Ap,F,I},
    s::I1,
    a::I2,
) where {Ap,F,I,I1<:Integer,I2<:Integer} = RLCore.forward(L.approximator, s, a)

function extract_sa(t::Traces{Tr}) where {Tr}
    # TODO: Delete this when RLTrajectories.jl is fixed
    # Hard coded to deal with index type instability in RLTrajectories.jl
    S = t.traces[1][:next_state][1] # pull 'next_state' from trajectory, e.g. latest state pushed
    A = t.traces[2][:next_action][1]
    return (S, A)
end


function _optimise!(
    n::I1,
    γ::F,
    app::TabularApproximator{2,Ar,O},
    s::I2,
    s_next::I2,
    a::I3,
    r::F,
) where {I1<:Number,I2<:Number,I3<:Number,Ar<:AbstractArray,F<:AbstractFloat,O}
        α = app.optimizer.eta
        Q!(app, s, s_next, a, α, r, γ)
end

function RLBase.optimise!(L::TDLearnerSARS{Ap,F,I}, cache::RLCore.SRT, t::Traces{Tr}) where {Ap,F,I,Tr}
    # S, A, R, T = (t[x][1] for x in SART)
    S, A = extract_sa(t) # Remove this when the above line works without a performance hit
    R = cache.reward
    S_next = cache.state
    _optimise!(L.n, L.γ, L.approximator, S, S_next, A, R)
end

function RLBase.priority(L::TDLearnerSARS{Ap,F,I}, transition::Tuple{T}) where {Ap,F,I,T}
    s, a, r, d, s′ = transition
    γ, Q = L.γ, L.approximator
    if d
        Δ = (r - RLCore.forward(Q, s, a))
    else
        Δ = (r + γ * RLCore.forward(Q, s′) - RLCore.forward(Q, s, a))
    end
    Δ = [Δ]  # must be broadcastable in Flux.Optimise
    Flux.Optimise.apply!(Q.optimizer, (s, a), Δ)
    abs(Δ[])
end
