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

using ReinforcementLearning

import ReinforcementLearning: RLBase
using Random
import Base.push!
import Base.getindex
using DataStructures: CircularBuffer

# Patch for Agent -> QBasedPolicy
function RLBase.optimise!(agent::Agent, stage::PostActStage)
    optimise!(agent.policy, agent.trajectory)
end

function RLBase.optimise!(policy::QBasedPolicy, trajectory::Trajectory)
    for batch in trajectory.container
        optimise!(policy.learner, batch)
    end
end

RLBase.optimise!(agent::MultiAgentPolicy, stage::PreActStage) = nothing
RLBase.optimise!(agent::MultiAgentPolicy, stage::PostEpisodeStage) = nothing
RLBase.optimise!(agent::MultiAgentPolicy, stage::PreEpisodeStage) = nothing

const SART = (:state, :action, :reward, :terminal)

function RLBase.plan!(
    s::AIAPCEpsilonGreedyExplorer{<:Any,F},
    values,
    full_action_space,
) where {F<:AbstractFloat}
    # NOTE: use of legal_action_space_mask as full_action_space is a bit of a hack, won't work in other cases
    ϵ = get_ϵ(s)
    s.step[1] += 1
    if rand(s.rng) < ϵ
        return rand(s.rng, full_action_space)
    else
        max_vals = find_all_max(values)[2]
        return rand(s.rng, max_vals)
    end
end

# Handle CartesianIndex actions
function Base.push!(
    multiagent::MultiAgentPolicy,
    ::PostActStage,
    env::E,
    actions::CartesianIndex,
) where {E<:AbstractEnv}
    actions = Tuple(actions)
    Base.push!(multiagent, PostActStage(), env, actions)
end
