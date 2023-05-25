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
using StaticArrays



# Epsilon Greedy Explorer for AIAPC Zoo
# Note: get_ϵ function in RLCore takes: 600.045 ns (6 allocations: 192 bytes)
# This one has: 59.003 ns (1 allocation: 16 bytes)
# Well worth looking into optimizations for RLCore
# TODO evaluate performance cost of checking all values for max, perhaps only do this in the beginning?
mutable struct AIAPCEpsilonGreedyExplorer{R,F<:AbstractFloat} <: AbstractExplorer
    β::F
    β_neg::F
    step::Int
    rng::R
    maximum_placeholder::Vector{F}
end

function AIAPCEpsilonGreedyExplorer(β::F) where {F<:AbstractFloat}
    AIAPCEpsilonGreedyExplorer{typeof(Random.GLOBAL_RNG),F}(β, β * -1, 1, Random.GLOBAL_RNG, F[1])
end

function get_ϵ(s::AIAPCEpsilonGreedyExplorer{<:Any,F}, step) where {F<:AbstractFloat}
    step_doubled = step * 2# FIXME TODO: This is a hack to get to 14% like AIAPC paper
    exp(s.β_neg * step_doubled)
end

get_ϵ(s::AIAPCEpsilonGreedyExplorer{<:Any,F}) where {F<:AbstractFloat} = get_ϵ(s, s.step)

function find_all_max_(x, maximum_placeholder)
    maximum!(maximum_placeholder, x)
    return findall(==(maximum_placeholder[1]), x)
end

function RLBase.plan!(
    s::AIAPCEpsilonGreedyExplorer{<:Any,F},
    values,
    full_action_space,
) where {F<:AbstractFloat}
    # NOTE: use of legal_action_space_mask as full_action_space is a bit of a hack, won't work in other cases
    ϵ = get_ϵ(s)
    s.step += 1
    if rand(s.rng) < ϵ
        return rand(s.rng, full_action_space)
    end
    max_vals = find_all_max_(values, s.maximum_placeholder)

    return rand(s.rng, max_vals)
end

# Patch for QBasedPolicy, not sure why NamedTuple dispatch is not working
RLBase.optimise!(p::QBasedPolicy, x::CircularArraySARTTraces) = optimise!(p.learner, x)

const SART = (:state, :action, :reward, :terminal)
