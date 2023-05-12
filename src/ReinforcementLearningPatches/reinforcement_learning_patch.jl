# The ReinforcementLearning.jl package is licensed under the MIT "Expat" License:

# > Copyright (c) 2018-2021: Johanni Brea.
# >
# > Permission is hereby granted, free of charge, to any person obtaining a copy
# > of this software and associated documentation files (the "Software"), to deal
# > in the Software without restriction, including without limitation the rights
# > to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# > copies of the Software, and to permit persons to whom the Software is
# > furnished to do so, subject to the following conditions:
# >
# > The above copyright notice and this permission notice shall be included in all
# > copies or substantial portions of the Software.
# >
# > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# > IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# > FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# > AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# > LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# > OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# > SOFTWARE.
# >

using ReinforcementLearningCore

using ReinforcementLearningBase
import ReinforcementLearningBase: RLBase
using ReinforcementLearningEnvironments
using Random

# Fudge an issue with non Int64 Ints in RL.jl
using Base
Base.convert(::Type{Int64}, a::Int64) = Int64(a)

### Patch modified from https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/v0.10.1/src/ReinforcementLearningCore/src/policies/q_based_policies/learners/approximators/tabular_approximator.jl
### To support smaller ints / floats
# (app::TabularQApproximator)(s::Int64) = @views app.table[:, s]
# (app::TabularQApproximator)(s::Int64, a::Int64) = app.table[a, s]

# add missing update! method for smaller Int types
# function RLBase.optimise!(
#     app::TabularQApproximator,
#     correction::Pair{Tuple{Int64,Int64},Float64},
# )
#     (s, a), e = correction
#     x = @view app.table[a, s]
#     x̄ = @view Float64[e][1]

#     Flux.Optimise.update!(app.optimizer, x, x̄)
# end

# function RLBase.optimise!(app::TabularQApproximator, correction::Pair{Int64,Vector{Float64}})
#     s, errors = correction
#     x = @view app.table[:, s]
#     Flux.Optimise.update!(app.optimizer, x, errors)
# end


# Epsilon Greedy Explorer for AIAPC Zoo
# Note: get_ϵ function in RLCore takes: 600.045 ns (6 allocations: 192 bytes)
# This one has: 59.003 ns (1 allocation: 16 bytes)
# Well worth looking into optimizations for RLCore
# TODO evaluate performance cost of checking all values for max, perhaps only do this in the beginning?
mutable struct AIAPCEpsilonGreedyExplorer{R} <: AbstractExplorer
    β::Float64
    β_neg::Float64
    step::Int
    rng::R
end

function AIAPCEpsilonGreedyExplorer(β::Float64)
    AIAPCEpsilonGreedyExplorer{typeof(Random.GLOBAL_RNG)}(β, β * -1, 1, Random.GLOBAL_RNG)
end

function get_ϵ(s::AIAPCEpsilonGreedyExplorer{<:Any}, step)
    exp(s.β_neg * step)
end

get_ϵ(s::AIAPCEpsilonGreedyExplorer{<:Any}) = get_ϵ(s, s.step)

function RLBase.plan!(s::AIAPCEpsilonGreedyExplorer{<:Any}, values::T, full_action_space) where {T}
    ϵ = get_ϵ(s)
    s.step += 1
    if rand(s.rng) < ϵ
        return rand(s.rng, full_action_space)
    end
    max_vals = find_all_max(values)[2]

    return rand(s.rng, max_vals)
end

# Patch for QBasedPolicy, not sure why NamedTuple dispatch is not working
RLBase.optimise!(p::QBasedPolicy, x::CircularArraySARTTraces) = optimise!(p.learner, x)

const SART = (:state, :action, :reward, :terminal)
