using Chain
using ReinforcementLearningCore, ReinforcementLearningBase
using DataFrames

function profit_gain(π_hat, env)
    π_N, π_M = extract_profit_vars(env)
    (mean(π_hat) - π_N) / (π_M - π_N)
end

struct AIAPCSummary
    α::Float64
    β::Float64
    is_converged::Vector{Bool}
    avg_profit::Vector{Float64}
    iterations_until_convergence::Vector{Int64}
end

function extract_profit_vars(env::AIAPCEnv)
    p_Bert_nash_equilibrium = env.p_Bert_nash_equilibrium
    p_monop_opt = env.p_monop_opt
    competition_params = env.competition_solution.params

    π_N = π(p_Bert_nash_equilibrium, p_Bert_nash_equilibrium, competition_params)[1]
    π_M = π(p_monop_opt, p_monop_opt, competition_params)[1]
    return (π_N, π_M)
end

economic_summary(e::RLCore.Experiment) = economic_summary(e.env, e.hook)

function economic_summary(env::AbstractEnv, hook::AbstractHook)
    convergence_threshold = env.convergence_threshold
    iterations_until_convergence = Int64[
        hook[player][2].iterations_until_convergence for player in [Symbol(1), Symbol(2)]
    ]

    avg_profit = Float64[]
    is_converged = Bool[]

    for i in (Symbol(1), Symbol(2))
        @chain hook[i][1].rewards[(end-convergence_threshold):end] begin
            push!(avg_profit, mean(_))
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

function extract_sim_results(exp_list::Vector{AIAPCSummary})
    α_result = [ex.α for ex in exp_list if !(ex isa Exception)]
    β_result = [ex.β for ex in exp_list if !(ex isa Exception)]
    iterations_until_convergence =
        [ex.iterations_until_convergence[1] for ex in exp_list if !(ex isa Exception)]

    avg_profit_result = [ex.avg_profit[1] for ex in exp_list if !(ex isa Exception)]

    df = DataFrame(
        α = α_result,
        β = β_result,
        π_bar = avg_profit_result,
        iterations_until_convergence = iterations_until_convergence,
    )
    return df
end
