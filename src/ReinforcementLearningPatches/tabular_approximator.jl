using ReinforcementLearningBase

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

(app::TabularVApproximator)(s::I) where {I <: Integer} = @views app.table[s]

(app::TabularQApproximator)(s::I) where {I <: Integer} = @views app.table[:, s]
(app::TabularQApproximator)(s::I1, a::I2) where {I1 <: Integer, I2 <: Integer} = app.table[a, s]

function RLBase.optimise!(app::TabularVApproximator, correction::Pair{I,F}) where {I <: Integer, F <: AbstractFloat}
    s, e = correction
    x = @view app.table[s]
    x̄ = @view [e][1]
    Flux.Optimise.update!(app.optimizer, x, x̄)
end

function RLBase.optimise!(app::TabularQApproximator, correction::Pair{Tuple{I1,I2},F}) where {I1 <: Integer, I2 <: Integer, F <: AbstractFloat}
    (s, a), e = correction
    x = @view app.table[a, s]
    x̄ = @view [e][1]
    Flux.Optimise.update!(app.optimizer, x, x̄)
end

function RLBase.optimise!(app::TabularQApproximator, correction::Pair{I,Vector{F}}) where {I <: Integer, F <: AbstractFloat}
    s, errors = correction
    x = @view app.table[:, s]
    Flux.Optimise.update!(app.optimizer, x, errors)
end
