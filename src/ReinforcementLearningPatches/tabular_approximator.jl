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
struct TabularApproximator{O}
    table::Matrix{Float64}
    optimizer::O
    function TabularApproximator(table::Matrix{Float64}, opt::O) where {O}
        n = ndims(table)
        n <= 2 || throw(ArgumentError("the dimension of table must be <= 2"))
        new{O}(table, opt)
    end
end

const TabularQApproximator = TabularApproximator{2}

TabularQApproximator(; n_state, n_action, init = 0.0, opt = InvDecay(1.0)) =
    TabularApproximator(fill(init, n_action, n_state), opt)

_get_qapproximator(table::Matrix{Float64}, s::I) where {I<:Integer} = @views table[:, s]
_get_qapproximator(table::Matrix{Float64}, s::I1, a::I2) where {I1<:Integer,I2<:Integer} = Float64(table[a, s])
_get_qapproximator_as_view(table::Matrix{Float64}, s::I) where {I<:Integer} = @view table[:, s]
_get_qapproximator_as_view(table::Matrix{Float64}, s::I1, a::I2) where {I1<:Integer,I2<:Integer} = @view table[a, s]
    
RLCore.estimate_reward(table::Matrix{Float64}, s::I) where {I<:Integer} = _get_qapproximator(table, s)
RLCore.estimate_reward(table::Matrix{Float64}, s::I1, a::I2) where {I1<:Integer,I2<:Integer} = _get_qapproximator(table, s, a)

function RLBase.optimise!(
    table::Matrix{Float64},
    optimizer::Flux.Optimiser.Descent,
    s::Int64,
    a::Int64,
    e::Float64,
)
    x = _get_qapproximator(table, s, a)
    x = @view [x][1]
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
