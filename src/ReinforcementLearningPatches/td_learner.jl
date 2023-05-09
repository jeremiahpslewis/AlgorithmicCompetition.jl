export TDLearner

using LinearAlgebra: dot
using Distributions: pdf
using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningTrajectories: Trajectory

Base.@kwdef struct TDLearner{A,F<:AbstractFloat,I<:Integer} <: AbstractLearner
    approximator::A
    γ::F = 1.0
    method::Symbol
    n::I = 0
end

(L::TDLearner)(env::E) where {E<:AbstractEnv} = L.approximator(state(env))
(L::TDLearner)(s) = L.approximator(s)
(L::TDLearner)(s, a) = L.approximator(s, a)

function RLBase.optimise!(L::TDLearner, t)
    _optimise!(L, t)
end

# NOTE: Indexing is hard-coded due to RLTrajectories type instability
function _optimise!(L::TDLearner, t)
    S = t.container.traces[1][:state]
    A = t.container.traces[2][:action]
    R = t.container.traces[3]
    T = t.container.traces[4]

    n, γ, Q = L.n, L.γ, L.approximator
    G = 0.0
    for i = 1:min(n + 1, length(R))
        G = R + γ * G
        s, a = S[end-i], A[end-i]
        RLBase.optimise!(Q, (s, a) => Q(s, a) - G)
    end
end

function RLBase.priority(L::TDLearner, transition::Tuple)
    if L.method == :SARS
        s, a, r, d, s′ = transition
        γ, Q = L.γ, L.approximator
        Δ = d ? (r - Q(s, a)) : (r + γ^(L.n + 1) * maximum(Q(s′)) - Q(s, a))
        Δ = [Δ]  # must be broadcastable in Flux.Optimise
        Flux.Optimise.apply!(Q.optimizer, (s, a), Δ)
        abs(Δ[])
    else
        @error "unsupported method"
    end
end
