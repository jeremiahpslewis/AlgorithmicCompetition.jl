export TDLearner

using LinearAlgebra: dot
using Distributions: pdf
using ReinforcementLearningBase
using ReinforcementLearningCore
import ReinforcementLearningCore: RLCore
using ReinforcementLearningTrajectories: Trajectory

Base.@kwdef struct TDLearner{A,F<:AbstractFloat,I<:Integer} <: AbstractLearner
    approximator::A
    γ::F = 1.0
    method::Symbol
    n::I = 0
end

RLCore.estimate_reward(L::TDLearner{A,F,I}, env::E) where {A,F,I,E<:AbstractEnv} = RLCore.estimate_reward(L.approximator, state(env))
RLCore.estimate_reward(L::TDLearner{A,F,I}, s) where {A,F,I} = RLCore.estimate_reward(L.approximator, s)
RLCore.estimate_reward(L::TDLearner{A,F,I}, s, a) where {A,F,I} = RLCore.estimate_reward(L.approximator, s, a)

function RLBase.optimise!(L::TDLearner, t)
    _optimise!(L, t)
end

# NOTE: Indexing is hard-coded due to RLTrajectories type instability
function _optimise!(L::TDLearner{Ap,F,I}, t::Tr) where {Ap,F,I,Tr<:Traces}
    S = t.traces[1][:state][1]
    A = t.traces[2][:action][1]
    R = t.traces[3][1]
    T = t.traces[4][1]

    n, γ, Q = L.n, L.γ, L.approximator
    G = 0.0
    for i = 1:min(n + 1, length(R))
        G = R + γ * G
        s, a = S[end-i], A[end-i]
        RLBase.optimise!(Q, s, a, RLCore.estimate_reward(Q, s, a) - G)
    end
end

function RLBase.priority(L::TDLearner, transition::Tuple)
    if L.method == :SARS
        s, a, r, d, s′ = transition
        γ, Q = L.γ, L.approximator
        Δ = d ? (r - RLCore.estimate_reward(Q, s, a)) : (r + γ^(L.n + 1) * maximum(RLCore.estimate_reward(Q, s′)) - RLCore.estimate_reward(Q, s, a))
        Δ = [Δ]  # must be broadcastable in Flux.Optimise
        Flux.Optimise.apply!(Q.optimizer, (s, a), Δ)
        abs(Δ[])
    else
        @error "unsupported method"
    end
end
