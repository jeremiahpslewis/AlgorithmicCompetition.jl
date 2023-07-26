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

RLCore.forward(L::TDLearnerSARS{Ap,F,I}, s::I1) where {Ap,F,I<:Integer,I1<:Integer} =
    RLCore.forward(L.approximator, s)
RLCore.forward(
    L::TDLearnerSARS{Ap,F,I},
    s::I1,
    a::I2,
) where {Ap,F,I,I1<:Integer,I2<:Integer} = RLCore.forward(L.approximator, s, a)

function extract_sars(t::Traces{Tr}) where {Tr}
    # TODO: Delete this when RLTrajectories.jl is fixed
    # Hard coded to deal with index type instability in RLTrajectories.jl
    S = t.traces[1][:state][1]
    S_next = t.traces[1][:next_state][1]
    A = t.traces[2][1] # action
    R = t.traces[3][1] # reward
    return (S, A, R, S_next)
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

function RLBase.optimise!(L::TDLearnerSARS{Ap,F1,I}, t::@NamedTuple{state::I1, next_state::I1, action::I2, reward::F2, terminal::Bool}) where {Ap,F1<:AbstractFloat,I,I1<:Number,I2<:Number,I3<:Number,Ar<:AbstractArray,F2<:AbstractFloat}
    _optimise!(L.n, L.γ, L.approximator, t.state, t.next_state, t.action, t.reward)
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
