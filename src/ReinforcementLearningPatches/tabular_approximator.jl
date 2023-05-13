using ReinforcementLearningBase
import ReinforcementLearningCore: RLCore
using Flux

"""
    TabularApproximator(table<:AbstractArray, opt)

For `table` of 1-d, it will serve as a state value approximator.
For `table` of 2-d, it will serve as a state-action value approximator.

!!! warning
    For `table` of 2-d, the first dimension is action and the second dimension is state.
"""
# TODO: add back missing AbstractApproximator
struct TabularApproximator{N,R,O}
    # R is the reward type
    table::AbstractArray{R,N}
    optimizer::O
    function TabularApproximator(table::AbstractArray{R,N}, opt::O) where {N,R,O}
        n = ndims(table)
        n <= 2 || throw(ArgumentError("the dimension of table must be <= 2"))
        new{N,R,O}(table, opt)
    end
end

TabularVApproximator(; n_state, init = 0.0, opt = InvDecay(1.0)) =
    TabularApproximator{typeof(init)}(fill(init, n_state), opt)

TabularQApproximator(; n_state, n_action, init = 0.0, opt = InvDecay(1.0)) =
    TabularApproximator{typeof(init)}(fill(init, n_action, n_state), opt)

RLCore.estimate_reward(app::TabularApproximator{1,R,O}, s::I) where {R,O,I} = @views app.table[s]

RLCore.estimate_reward(app::TabularApproximator{2,R,O}, s::I) where {R,O,I<:Integer} = @views app.table[:, s]
RLCore.estimate_reward(app::TabularApproximator{2,R,O}, s::I1, a::I2) where {R,O,I1<:Integer,I2<:Integer} = @views app.table[a, s]

function RLBase.optimise!(
    app::TabularApproximator{1,R,O},
    correction::Pair{I,F},
) where {R,O,I<:Integer,F<:AbstractFloat}
    s, e = correction
    x = @view app.table[s]
    x̄ = @view [e][1]
    Flux.Optimise.update!(app.optimizer, x, x̄)
end

function RLBase.optimise!(
    app::TabularApproximator{2,R,O},
    correction::Pair{Tuple{I1,I2},F},
) where {R,O,I1<:Integer,I2<:Integer,F<:AbstractFloat}
    (s, a), e = correction
    x = @view app.table[a, s]
    x̄ = @view [e][1]
    Flux.Optimise.update!(app.optimizer, x, x̄)
end

function RLBase.optimise!(
    app::TabularApproximator{2,R,O},
    correction::Pair{I,Vector{F}},
) where {R,O,I<:Integer,F<:AbstractFloat}
    s, errors = correction
    x = @view app.table[:, s]
    Flux.Optimise.update!(app.optimizer, x, errors)
end
