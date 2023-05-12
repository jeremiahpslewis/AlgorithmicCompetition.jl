using Chain
using ReinforcementLearningCore, ReinforcementLearningBase

profit_measure(π_hat::Vector{Float64}, π_N, π_M) = (mean(π_hat) - π_N) / (π_M - π_N)

struct AIAPCSummary
    α::Float64
    β::Float64
    is_converged::Vector{Bool}
    avg_profit::Vector{Float64}
    iterations_until_convergence::Vector{Int32}
end

function economic_summary(e::RLCore.Experiment)
    convergence_threshold = e.env.convergence_threshold
    iterations_until_convergence = [
        e.hook[player][2].iterations_until_convergence for player in [Symbol(1), Symbol(2)]
    ]

    π_N = e.env.profit_function(fill(e.env.p_Bert_nash_equilibrium, 2))[1]
    π_M = e.env.profit_function(fill(e.env.p_monop_opt, 2))[1]

    avg_profit = Float64[]
    is_converged = Bool[]

    for i in [Symbol(1), Symbol(2)]
        @chain e.hook[i][1].rewards[(end-convergence_threshold):end] begin
            push!(avg_profit, profit_measure(_, π_N, π_M))
        end

        push!(is_converged, e.hook[i][2].is_converged)
    end

    return AIAPCSummary(
        e.env.α,
        e.env.β,
        is_converged,
        avg_profit,
        iterations_until_convergence,
    )
end
