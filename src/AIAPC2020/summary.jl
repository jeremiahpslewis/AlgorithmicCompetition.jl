using Chain
using ReinforcementLearningCore, ReinforcementLearningBase

profit_measure(π_hat, π_N, π_M) = (mean(π_hat) - π_N) / (π_M - π_N)

struct AIAPCSummary
    α::Float64
    β::Float64
    is_converged::Vector{Bool}
    avg_profit::Vector{Float64}
    iterations_until_convergence::Vector{Int32}
end

function extract_profit_vars(env)
    p_N = env.p_Bert_nash_equilibrium
    p_M = env.p_monop_opt
    π_N = π(fill(p_N, 2), env.competition_solution.params)[1]
    π_M = π(fill(p_M, 2), env.competition_solution.params)[1]
    return (π_N, π_M)
end

economic_summary(e::RLCore.Experiment) = economic_summary(e.env, e.hook)

function economic_summary(env, hook)
    convergence_threshold = env.convergence_threshold
    iterations_until_convergence = Int32[
        hook[player][2].iterations_until_convergence for player in [Symbol(1), Symbol(2)]
    ]

    π_N, π_M = extract_profit_vars(env)

    avg_profit = Float64[]
    is_converged = Bool[]

    for i in [Symbol(1), Symbol(2)]
        @chain hook[i][1].rewards[(end-convergence_threshold):end] begin
            push!(avg_profit, profit_measure(_, π_N, π_M))
        end

        push!(is_converged, hook[i][2].is_converged)
    end

    return AIAPCSummary(
        env.α,
        env.β,
        is_converged,
        avg_profit,
        iterations_until_convergence,
    )
end
