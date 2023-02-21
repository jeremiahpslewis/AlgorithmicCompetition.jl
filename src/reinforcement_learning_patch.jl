using ReinforcementLearningZoo
using ReinforcementLearning

# Work around NoOp simultaneous -> sequential transformation
# From https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/2e1de3e5b6b8224f50b3d11bba7e1d2d72c6ef7c/src/ReinforcementLearningZoo/src/algorithms/tabular/td_learner.jl#L130
function ReinforcementLearningZoo._update!(
    L::TDLearner,
    ::TabularQApproximator,
    ::Val{:SARS},
    t::VectorSARTTrajectory,
    ::PreActStage,
)
    S = t[:state]
    A = t[:action]
    R = t[:reward]

    n, γ, Q = L.n, L.γ, L.approximator

    if length(R) >= n + 1
        s, a, s′ = S[end-n-1], A[end-n-1], S[end]
        if !(a isa NoOp)
            G =
                discount_rewards_reduced(@view(R[end-n:end]), γ) +
                γ^(n + 1) * maximum(Q(s′))
            ReinforcementLearning.update!(Q, (s, a) => Q(s, a) - G)
        end
    end
end

# Work around NoOp simultaneous -> sequential transformation
# From https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/2e1de3e5b6b8224f50b3d11bba7e1d2d72c6ef7c/src/ReinforcementLearningZoo/src/algorithms/tabular/td_learner.jl#L74
function ReinforcementLearningZoo._update!(
    L::TDLearner,
    ::TabularQApproximator,
    ::Val{:SARS},
    t::VectorSARTTrajectory,
    ::PostEpisodeStage,
)
    S, A, R, T = [t[x] for x in SART]
    n, γ, Q = L.n, L.γ, L.approximator
    G = 0.0

    for i = 1:min(n + 1, length(R))
        G = R[end-i+1] + γ * G
        s, a = S[end-i], A[end-i]
        if !(a isa NoOp)
            ReinforcementLearning.update!(Q, (s, a) => Q(s, a) - G)
        end
    end
end


# Support Int < 64...
(app::TabularQApproximator)(s::Int16) = @views app.table[:, s]
(app::TabularQApproximator)(s::Int16, a::Int8) = app.table[a, s]
