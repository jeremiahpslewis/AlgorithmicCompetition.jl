function RLCore.forward(app::TabularApproximator{2,R,O},
    s::I1, a::I2) where {R<:AbstractArray,O,I1<:Integer,I2<:Integer}
    RLCore.forward(app, s, a.price_index)
end
    
function RLBase.optimise!(app::TabularApproximator{2,R,O}, s_a::Tuple{I1,I2},
e::F) where {R<:AbstractArray,O,I1<:Integer,I2<:Integer,F<:AbstractFloat}
    RLBase.optimise!(app, (s_a[1], s_a[2].price_index), e)
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
