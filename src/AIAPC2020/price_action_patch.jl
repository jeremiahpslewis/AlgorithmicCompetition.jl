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
