# Source: https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/701d2f976bc40d10f0b5b751e237f3ce01b228ff/src/ReinforcementLearningFarm/src/algorithms/explorers/epsilon_speedy_explorer.jl
using ReinforcementLearning
import ReinforcementLearning: RLBase

using FillArrays: Trues
using Random
using Distributions: Categorical
using Base

"""
    EpsilonSpeedyExplorer2(β::Float64)

`EpsilonSpeedyExplorer2` is an explorer that selects the action with the maximum value with probability `1 - ϵ` and selects a random action with probability `ϵ`.
The probability of selecting a random action is given by `exp(β * -t)`, where `t` is the number of times `plan!` has been called.
`EpsilonSpeedyExplorer2` differs from `EpsilonGreedyExplorer` in that it uses the `exp` function to calculate the probability of selecting a random action over the full range of `t` and only accepts one argument, `β`.

"""
struct EpsilonSpeedyExplorer2{R} <: AbstractExplorer
    β::Float64
    β_neg::Float64
    min_ϵ::Float64 # minimum epsilon value to ensure exploration never goes to zero (for alternate spec)
    step::Base.RefValue{Int}
    rng::R
end

function EpsilonSpeedyExplorer2(β::Float64; min_ϵ::Float64 = 0.0)
    EpsilonSpeedyExplorer2{typeof(Random.GLOBAL_RNG)}(
        β,
        β * -1,
        min_ϵ,
        Ref(1),
        Random.GLOBAL_RNG,
    )
end

function get_ϵ(s::EpsilonSpeedyExplorer2)
    max(exp(s.β_neg * s.step[]), s.min_ϵ)
end

"""
    RLBase.plan!(s::EpsilonSpeedyExplorer2, values; step) where T

!!! note
    If multiple values with the same maximum value are found.
    Then a random one will be returned when `is_break_tie==true`.

    `NaN` will be filtered unless all the values are `NaN`.
    In that case, a random one will be returned.
"""
function RLBase.plan!(s::EpsilonSpeedyExplorer2{R}, values::A) where {I<:Real, A<:AbstractArray{I}, R<:Random.AbstractRNG}
    ϵ = get_ϵ(s)
    s.step[] += 1
    rand(s.rng) >= ϵ ? findmax(values)[2] : rand(s.rng, 1:length(values))
end

RLBase.plan!(s::EpsilonSpeedyExplorer2{R}, x::A, mask::Trues) where {I<:Real, A<:AbstractArray{I}, R<:Random.AbstractRNG} = RLBase.plan!(s, x)

function RLBase.plan!(s::EpsilonSpeedyExplorer2{R}, values::A, mask::M) where {I<:Real, A<:AbstractArray{I}, M<:Union{BitVector, Vector{Bool}}, R<:Random.AbstractRNG}
    ϵ = get_ϵ(s)
    s.step[] += 1
    # NOTE: takes first max element, doesn't break ties randomly
    rand(s.rng) >= ϵ ? RLCore.findmax_masked(values, mask)[2] : rand(s.rng, findall(mask))
end

"""
    prob(s::EpsilonGreedyExplorer, values) -> Categorical
    prob(s::EpsilonGreedyExplorer, values, mask) -> Categorical

Return the probability of selecting each action given the estimated `values` of each action.
"""
function RLBase.prob(s::EpsilonSpeedyExplorer2, values::A) where {I<:Real, A<:AbstractArray{I}}
    ϵ, n = get_ϵ(s), length(values)
    probs = fill(ϵ / n, n)
    probs[findmax(values)[2]] += 1 - ϵ
    Categorical(probs; check_args=false)
end

function RLBase.prob(s::EpsilonSpeedyExplorer2, values::A, action::Integer) where {I<:Real, A<:AbstractArray{I}}
    ϵ, n = get_ϵ(s), length(values)
    if action == findmax(values)[2]
        ϵ / n + 1 - ϵ
    else
        ϵ / n
    end
end

function RLBase.prob(s::EpsilonSpeedyExplorer2, values::A, mask::M) where {I<:Real, A<:AbstractArray{I}, M<:Union{BitVector, Vector{Bool}}}
    ϵ, n = get_ϵ(s), length(values)
    probs = zeros(n)
    probs[mask] .= ϵ / sum(mask)
    probs[RLCore.findmax_masked(values, mask)[2]] += 1 - ϵ
    Categorical(probs; check_args=false)
end
