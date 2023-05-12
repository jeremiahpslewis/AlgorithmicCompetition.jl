using ReinforcementLearningBase
import ReinforcementLearningBase: RLBase
import ReinforcementLearningCore: RLCore

"""
    TabularApproximator(table<:AbstractArray, opt)

For `table` of 1-d, it will serve as a state value approximator.
For `table` of 2-d, it will serve as a state-action value approximator.

!!! warning
    For `table` of 2-d, the first dimension is action and the second dimension is state.
"""
# TODO: add back missing AbstractApproximator
struct TabularApproximator{N,T<:AbstractArray,O}
    table::T
    optimizer::O
    function TabularApproximator(table::T, opt::O) where {T<:AbstractArray,O}
        n = ndims(table)
        n <= 2 || throw(ArgumentError("the dimension of table must be <= 2"))
        new{n,T,O}(table, opt)
    end
end

const TabularVApproximator = TabularApproximator{1}
const TabularQApproximator = TabularApproximator{2}

TabularVApproximator(; n_state, init = 0.0, opt = InvDecay(1.0)) =
    TabularApproximator(fill(init, n_state), opt)
TabularQApproximator(; n_state, n_action, init = 0.0, opt = InvDecay(1.0)) =
    TabularApproximator(fill(init, n_action, n_state), opt)

_get_vapproximator(table::Matrix{Float32}, s::I) where {I<:Integer} = @views table[s]
_get_qapproximator(table::Matrix{Float32}, s::I) where {I<:Integer} = @views table[:, s]
_get_qapproximator(table::Matrix{Float32}, s::I1, a::I2) where {I1<:Integer,I2<:Integer} = @views table[a, s]
_get_qapproximator_as_view(table::Matrix{Float32}, s::I) where {I<:Integer} = @view table[:, s]
_get_qapproximator_as_view(table::Matrix{Float32}, s::I1, a::I2) where {I1<:Integer,I2<:Integer} = @view table[a, s]
    
RLCore.estimate_reward(app::TabularVApproximator, table::Matrix{Float32}, s::I) where {I<:Integer} = _get_vapproximator(table, s)
RLCore.estimate_reward(app::TabularQApproximator, table::Matrix{Float32}, s::I) where {I<:Integer} = _get_qapproximator(table, s)
RLCore.estimate_reward(app::TabularQApproximator, table::Matrix{Float32}, s::I1, a::I2) where {I1<:Integer,I2<:Integer} = _get_qapproximator(table, s, a)

function RLBase.optimise!(
    app::TabularVApproximator,
    s::I,
    e::F,
) where {I<:Integer,F<:AbstractFloat}
    s, e = correction
    x = _get_vapproximator(app.table, s)
    x̄ = @view [e][1]
    Flux.Optimise.update!(app.optimizer, x, x̄)
end

function RLBase.optimise!(
    app::TabularQApproximator,
    s::I1,
    a::I2,
    e::F,
) where {I1<:Integer,I2<:Integer,F<:AbstractFloat}
    x = _get_qapproximator_as_view(app.table, s, a)
    x̄ = @view [e][1]
    Flux.Optimise.update!(app.optimizer, x, x̄)
end

function RLBase.optimise!(
    app::TabularQApproximator,
    s::I,
    errors::Vector{F},
) where {I<:Integer,F<:AbstractFloat}
    x = _get_qapproximator_as_view(app.table, s)
    Flux.Optimise.update!(app.optimizer, x, errors)
end
