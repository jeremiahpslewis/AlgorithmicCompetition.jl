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
struct TabularApproximator{N,A,O}
    table::A
    optimizer::O
    function TabularApproximator(table::A, opt::O) where {A<:AbstractArray,O}
        n = ndims(table)
        n <= 2 || throw(ArgumentError("the dimension of table must be <= 2"))
        new{n,A,O}(table, opt)
    end
end

TabularVApproximator(; n_state, init = 0.0, opt = InvDecay(1.0)) =
    TabularApproximator(fill(init, n_state), opt)

TabularQApproximator(; n_state, n_action, init = 0.0, opt = InvDecay(1.0)) =
    TabularApproximator(fill(init, n_action, n_state), opt)



RLCore.forward(
    app::TabularApproximator{1,R,O},
    s::I,
) where {R<:AbstractArray,O,I<:Integer} = @views app.table[s]

RLCore.forward(
    app::TabularApproximator{2,R,O},
    s::I,
) where {R<:AbstractArray,O,I<:Integer} = @views app.table[:, s]
RLCore.forward(
    app::TabularApproximator{2,R,O},
    s::I1,
    a::I2,
) where {R<:AbstractArray,O,I1<:Integer,I2<:Integer} = @views app.table[a, s]

function RLBase.optimise!(
    app::TabularApproximator{1,R,O},
    s::I,
    e::F,
) where {R<:AbstractArray,O,I<:Integer,F<:AbstractFloat}
    x = @view app.table[s]
    x̄ = @view [e][1]
    Flux.Optimise.update!(app.optimizer, x, x̄)
end

function RLBase.optimise!(
    app::TabularApproximator{2,R,O},
    s_a::Tuple{I1,I2},
    e::F,
) where {R<:AbstractArray,O,I1<:Integer,I2<:Integer,F<:AbstractFloat}
    (s, a) = s_a
    x = @view app.table[a, s]
    x̄ = @view [e][1]
    Flux.Optimise.update!(app.optimizer, x, x̄)
end

function RLBase.optimise!(
    app::TabularApproximator{2,R,O},
    s::I,
    errors::Vector{F},
) where {R<:AbstractArray,O,I<:Integer,F<:AbstractFloat}
    x = @view app.table[:, s]
    Flux.Optimise.update!(app.optimizer, x, errors)
end
    
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
        max_vals = RLCore.find_all_max(values)[2]

        return Int8(rand(s.rng, max_vals))
    end
end
