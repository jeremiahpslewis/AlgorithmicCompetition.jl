Q(app::TabularApproximator, s, a) = RLCore.forward(app, s, a)
Q(app::TabularApproximator, s) = RLCore.forward(app, s)

function Q!(
    app::TabularApproximator,
    s::I1,
    s_plus_one::I2,
    a::I3,
    α::F1,
    π_::F2,
    γ::Float64,
) where {I1<:Integer,I2<:Integer,I3<:Integer,F1<:AbstractFloat,F2<:AbstractFloat}
    # Q-learning formula according to https://github.com/JuliaPOMDP/TabularTDLearning.jl/blob/25c4d3888e178c51ed1ff448f36b0fcaf7c1d8e8/src/q_learn.jl#LL63C26-L63C95
    q_value_updated = α * (π_ + γ * maximum(Q(app, s_plus_one)) - Q(app, s, a))
    app.table[a, s] += q_value_updated
    return Q(app, s, a)
end
