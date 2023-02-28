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

using ReinforcementLearningZoo
using ReinforcementLearning
using Random

# Reduce allocations
# From https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/2e1de3e5b6b8224f50b3d11bba7e1d2d72c6ef7c/src/ReinforcementLearningZoo/src/algorithms/tabular/td_learner.jl#L130
function _update!(
    L::TDLearner,
    ::TabularQApproximator,
    ::Val{:SARS},
    t::VectorSARTTrajectory,
    ::PostEpisodeStage,
)
    S, A, R, T = (t[x] for x in SART)
    n, γ, Q = L.n, L.γ, L.approximator
    G = 0.0
    for i in 1:min(n + 1, length(R))
        G = R[end-i+1] + γ * G
        s, a = S[end-i], A[end-i]
        update!(Q, (s, a) => Q(s, a) - G)
    end
end

### Patch modified from https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/v0.10.1/src/ReinforcementLearningCore/src/policies/q_based_policies/learners/approximators/tabular_approximator.jl
### To support smaller ints / floats
(app::TabularQApproximator)(s::Int16) = @views app.table[:, s]
(app::TabularQApproximator)(s::Int16, a::Int8) = app.table[a, s]

# add missing update! method for smaller Int types
function RLBase.update!(app::TabularQApproximator, correction::Pair{Tuple{Int16,Int8},Float64})
    (s, a), e = correction
    x = @view app.table[a, s]
    x̄ = @view Float64[e][1]

    Flux.Optimise.update!(app.optimizer, x, x̄)
end

function RLBase.update!(app::TabularQApproximator, correction::Pair{Int16,Vector{Float64}})
    s, errors = correction
    x = @view app.table[:, s]
    Flux.Optimise.update!(app.optimizer, x, errors)
end


function RLBase.update!(
    trajectory::VectorSARTTrajectory,
    policy::NamedPolicy,
    env::SequentialEnv,
    ::PostActStage,
)
    if env.current_player_idx == 1
        r = policy isa NamedPolicy ? reward(env.env, nameof(policy)) : reward(env)
        push!(trajectory[:reward], r)
        push!(trajectory[:terminal], is_terminated(env))
    end
end

function (agent::Agent)(stage::PreActStage, env::SequentialEnv, action::ReinforcementLearning.NoOp)
end

# Epsilon Greedy Explorer for AIAPC Zoo
mutable struct AIAPCEpsilonGreedyExplorer{R} <: AbstractExplorer
    β::Float32
    β_neg::Float32
    step::Int
    rng::R
end

function AIAPCEpsilonGreedyExplorer(
    β::Float32,
)
    AIAPCEpsilonGreedyExplorer{typeof(Random.GLOBAL_RNG)}(
        β,
        β * -1,
        1,
        Random.GLOBAL_RNG,
    )
end

function get_ϵ(s::AIAPCEpsilonGreedyExplorer{<:Any}, step)
    exp(s.β_neg * step)
end

get_ϵ(s::AIAPCEpsilonGreedyExplorer{<:Any}) = get_ϵ(s, s.step)

function (s::AIAPCEpsilonGreedyExplorer{<:Any})(values)
    ϵ = get_ϵ(s)
    s.step += 1
    rand(s.rng) >= ϵ ? rand(s.rng, find_all_max(values)[2]) : rand(s.rng, 1:length(values))
end

