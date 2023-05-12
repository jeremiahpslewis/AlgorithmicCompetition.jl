export TDLearner

using LinearAlgebra: dot
using Distributions: pdf
using ReinforcementLearningBase
using ReinforcementLearningCore
import ReinforcementLearningCore: RLCore
using ReinforcementLearningTrajectories: Trajectory

Base.@kwdef struct TDLearner{F<:AbstractFloat,I<:Integer} <: AbstractLearner
    approximator::TabularApproximator
    γ::F = 1.0
    method::Symbol
    n::I = 0
end

RLCore.estimate_reward(L::TDLearner, env::E) where {E<:AbstractEnv} = _get_qapproximator(L.approximator.table, RLBase.state(env))
RLCore.estimate_reward(L::TDLearner, s) = RLCore.estimate_reward(L.approximator.table, s)
RLCore.estimate_reward(L::TDLearner, s, a) = RLCore.estimate_reward(L.approximator.table, s, a)

function RLBase.optimise!(L::TDLearner, t)
    _optimise!(L, t)
    return
end

# NOTE: Indexing is hard-coded due to RLTrajectories type instability
function _optimise!(L::TDLearner, t::Tr) where {Tr<:Traces}
    S = t.traces[1][:state][1]
    A = t.traces[2][:action][1]
    R = t.traces[3][1]

    _optimise!(L, L.approximator.table, L.n, L.γ, S, A, R)
    return
end

function _optimise!(Q::TDLearner, table::Matrix{Float64}, n, γ, S, A, R)
    G = 0.0
    for i = 1:min(n + 1, length(R))
        G = R + γ * G
        s, a = S[end-i], A[end-i]
        RLBase.optimise!(Q.approximator.table, Q.approximator.optimizer, s, a, RLCore.estimate_reward(table, s, a) - G)
    end
    return
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
