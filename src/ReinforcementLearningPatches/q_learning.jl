Q(app::TabularApproximator, s, a) = RLCore.forward(app, s, a)
Q(app::TabularApproximator, s) = RLCore.forward(app, s)

function Q!(
    app::TabularApproximator,
    s::Int,
    s_plus_one::Int,
    a::Int,
    α::Float64,
    π_::Float64,
    δ::Float64,
)
    q_value_updated = (1 - α) * Q(app, s, a) + α * (π_ + δ * maximum(Q(app, s_plus_one)))
    app.table[a, s] = q_value_updated
    return q_value_updated
end
