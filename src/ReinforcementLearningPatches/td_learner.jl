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

RLCore.estimate_reward(L::TDLearnerSARS{Ap,F,I}, env::E) where {Ap,F,I,E<:AbstractEnv} = RLCore.estimate_reward(L.approximator, state(env))
RLCore.estimate_reward(L::TDLearnerSARS{Ap,F,I}, s::I1) where {Ap,F,I<:Integer,I1<:Integer} = RLCore.estimate_reward(L.approximator, s)
RLCore.estimate_reward(L::TDLearnerSARS{Ap,F,I}, s::I1, a::I2) where {Ap,F,I,I1<:Integer,I2<:Integer} = RLCore.estimate_reward(L.approximator, s, a)

function extract_sar(t::Tr) where {Tr<:Traces}
    # TODO: Delete this when RLTrajectories.jl is fixed
    # Hard coded to deal with index type instability in RLTrajectories.jl
    S = t.traces[1][:state][1]
    A = t.traces[2][:action][1]
    R = t.traces[3][1]
end

function RLBase.optimise!(L::TDLearnerSARS{Ap,F,I}, t::Tr) where {Ap,F,I,Tr<:Traces}
    # S, A, R, T = (t[x][1] for x in SART)
    S, A, R = extract_sar(t) # Remove this when the above line works without a performance hit
    n, γ, Q = L.n, L.γ, L.approximator
    G = 0.0
    for i = 1:min(n + 1, length(R))
        G = R + γ * G
        s, a = S[end-i], A[end-i]
        RLBase.optimise!(Q, (s, a) => RLCore.estimate_reward(Q, s, a) - G)
    end
end

function RLBase.priority(L::TDLearnerSARS{Ap,F,I}, transition::Tuple{T}) where {Ap,F,I,T}
        s, a, r, d, s′ = transition
        γ, Q = L.γ, L.approximator
        if d
            Δ = (r - RLCore.estimate_reward(Q, s, a))
        else
            Δ = (r + γ * RLCore.estimate_reward(Q, s′) - RLCore.estimate_reward(Q, s, a))
        end
        Δ = [Δ]  # must be broadcastable in Flux.Optimise
        Flux.Optimise.apply!(Q.optimizer, (s, a), Δ)
        abs(Δ[])
end
