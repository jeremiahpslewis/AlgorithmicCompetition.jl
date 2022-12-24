using Chain
using ReinforcementLearning
profit_measure(π_hat::Vector{Float64}, π_N, π_M) = (mean(π_hat) - π_N) / (π_M - π_N)

struct CalvanoSummary
    avg_profit::Vector{Float64}
end

function economic_summary(e::Experiment)
    convergence_threshold = e.env.env.convergence_threshold

    π_N = e.env.env.profit_function([p_Bert_nash_equilibrium, p_Bert_nash_equilibrium])[1]
    π_M = e.env.env.profit_function([p_monop_opt, p_monop_opt])[1]
    
    avg_profit = []

    for i in [1, 2]
        @chain e.hook.hooks[i][1].rewards[(end - convergence_threshold):end] begin
            push!(avg_profit, profit_measure(_, π_N, π_M))
        end
    end

    return CalvanoSummary(avg_profit)
end
