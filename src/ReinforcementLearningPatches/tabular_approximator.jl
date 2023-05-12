using ReinforcementLearningBase
import ReinforcementLearningBase: RLBase
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
struct TabularApproximator
    table::Matrix{Float64}
    optimizer::Descent
end

_get_qapproximator(table::Matrix{Float64}, s::I) where {I<:Integer} = @views table[:, s]
_get_qapproximator(table::Matrix{Float64}, s::I1, a::I2) where {I1<:Integer,I2<:Integer} = @views table[a, s]
_get_qapproximator_as_view(table::Matrix{Float64}, s::I) where {I<:Integer} = @view table[:, s]
_get_qapproximator_as_view(table::Matrix{Float64}, s::I1, a::I2) where {I1<:Integer,I2<:Integer} = @view table[a, s]
    
RLCore.estimate_reward(table::Matrix{Float64}, s::I) where {I<:Integer} = _get_qapproximator(table, s)
RLCore.estimate_reward(table::Matrix{Float64}, s::I1, a::I2) where {I1<:Integer,I2<:Integer} = _get_qapproximator(table, s, a)

function RLBase.optimise!(
    table::Matrix{Float64},
    optimizer::Descent,
    s::Int64,
    a::Int64,
    e::Float64,
)
    x = _get_qapproximator_as_view(table, s, a)
    x̄ = @view [e][1]
    Flux.Optimise.update!(optimizer, x, x̄)
    return
end

function RLBase.optimise!(
    app::TabularApproximator,
    s::I,
    errors::Vector{F},
) where {I<:Integer,F<:AbstractFloat}
    x = _get_qapproximator_as_view(app.table, s)
    Flux.Optimise.update!(app.optimizer, x, errors)
    return
end
