Q(app, s, a) = RLCore.forward(app, s, a)
Q(app, s) = RLCore.forward(app, s)

function Q!(app, s, s_plus_one, a, α, π, δ)
    q_value_updated = (1 - α) * Q(app, s, a) + α * (π + δ * maximum(Q(app, s_plus_one)))
    app.table[a, s] = q_value_updated
    return q_value_updated
end
