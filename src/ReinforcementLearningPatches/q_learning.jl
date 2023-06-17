Q(app::TabularApproximator, s, a) = RLCore.forward(app, s, a)
Q(app::TabularApproximator, s) = RLCore.forward(app, s)

function Q!(
    app::TabularApproximator,
    s::Int,
    s_plus_one::Int,
    a::Int,
    α::Float64,
    π_::Float64,
    γ::Float64,
)
    # Q-learning formula according to https://github.com/JuliaPOMDP/TabularTDLearning.jl/blob/25c4d3888e178c51ed1ff448f36b0fcaf7c1d8e8/src/q_learn.jl#LL63C26-L63C95
    q_value_updated = α * (r + γ * maximum(Q(app, s_plus_one)) - Q(app, s_plus_one, a))
    app.table[a, s] += q_value_updated
    return q_value_updated
end
