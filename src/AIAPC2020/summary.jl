using Chain
using ReinforcementLearning

profit_measure(π_hat::Vector{Float64}, π_N, π_M) = (mean(π_hat) - π_N) / (π_M - π_N)

struct AIAPCSummary
    α::Float32
    β::Float32
    is_converged::Vector{Bool}
    avg_profit::Vector{Float32}
    iterations_until_convergence::Int32
end

function economic_summary(e::ReinforcementLearningCore.Experiment)
    convergence_threshold = e.env.env.convergence_threshold
    iterations_until_convergence = e.hook.hooks[1][2].iterations_until_convergence

    π_N = e.env.env.profit_function(fill(e.env.env.p_Bert_nash_equilibrium, 2))[1]
    π_M = e.env.env.profit_function(fill(e.env.env.p_monop_opt, 2))[1]

    avg_profit = Float32[]
    is_converged = Bool[]

    for i in [1, 2]
        @chain e.hook.hooks[i][1].rewards[(end-convergence_threshold):end] begin
            push!(avg_profit, profit_measure(_, π_N, π_M))
        end

        push!(is_converged, e.hook.hooks[i][2].is_converged)
    end

    return AIAPCSummary(e.env.env.α,
        e.env.env.β,
        is_converged,
        avg_profit,
        iterations_until_convergence)
end
