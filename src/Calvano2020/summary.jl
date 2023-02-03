using Chain
using ReinforcementLearning
using JSON
using UUIDs

profit_measure(π_hat::Vector{Float64}, π_N, π_M) = (mean(π_hat) - π_N) / (π_M - π_N)

struct CalvanoSummary
    α::Float64
    β::Float64
    avg_profit::Vector{Float64}
end

function economic_summary(e::ReinforcementLearningCore.Experiment)
    convergence_threshold = e.env.env.convergence_threshold

    π_N = e.env.env.profit_function(fill(e.env.env.p_Bert_nash_equilibrium, 2))[1]
    π_M = e.env.env.profit_function(fill(e.env.env.p_monop_opt, 2))[1]

    avg_profit = Float64[]

    for i in [1, 2]
        @chain e.hook.hooks[i][1].rewards[(end - convergence_threshold):end] begin
            push!(avg_profit, profit_measure(_, π_N, π_M))
        end
    end

    return CalvanoSummary(e.env.env.α, e.env.env.β, avg_profit)
end
