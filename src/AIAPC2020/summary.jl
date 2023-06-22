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
    convergence_profit::Vector{Float64}
    iterations_until_convergence::Vector{Int64}
end

function extract_profit_vars(env::AIAPCEnv)
    p_Bert_nash_equilibrium = env.p_Bert_nash_equilibrium
    p_monop_opt = env.p_monop_opt
    competition_params = env.competition_params

    π_N = π(p_Bert_nash_equilibrium, p_Bert_nash_equilibrium, competition_params)[1]
    π_M = π(p_monop_opt, p_monop_opt, competition_params)[1]
    return (π_N, π_M)
end

economic_summary(e::RLCore.Experiment) = economic_summary(e.env, e.policy, e.hook)


function get_state_from_memory(env::AIAPCEnv)
    return get_state_from_prices(env, env.memory)
end

function get_state_from_prices(env::AIAPCEnv, memory)
    return env.state_space_lookup[memory[1].price_index, memory[2].price_index]
end

function get_prices_from_state(env::AIAPCEnv, state)
    prices = findall(x -> x == state, env.state_space_lookup)[1]
    return [env.price_options[prices[1]], env.price_options[prices[2]]]
end

function get_profit_from_state(env::AIAPCEnv, state)
    prices = get_prices_from_state(env, state)
    return AlgorithmicCompetition.π(prices[1], prices[2], env.competition_params)
end

function get_optimal_action(env::AIAPCEnv, policy::MultiAgentPolicy, last_observed_state)
    optimal_action_set = Int8[]
    for player_ in [Symbol(1), Symbol(2)]
        opt_act = argmax(
            policy[player_].policy.learner.approximator.table[:, last_observed_state],
        )
        push!(optimal_action_set, Int8(opt_act))
    end
    return optimal_action_set
end

function economic_summary(env::AbstractEnv, policy::MultiAgentPolicy, hook::AbstractHook)
    convergence_threshold = env.convergence_threshold
    iterations_until_convergence = Int64[
        hook[player][2].iterations_until_convergence for player in [Symbol(1), Symbol(2)]
    ]

    is_converged = Bool[]

    convergence_profit = [get_convergence_profit_from_env(env, policy)...]

    for i in (Symbol(1), Symbol(2))
        push!(is_converged, hook[i][2].is_converged)
    end

    return AIAPCSummary(
        env.α,
        env.β,
        is_converged,
        convergence_profit,
        iterations_until_convergence,
    )
end

function get_convergence_profit_from_env(env::AIAPCEnv, policy::MultiAgentPolicy)
    last_observed_state = get_state_from_memory(env)

    visited_states = [last_observed_state]

    for i = 1:100
        next_price_set = get_optimal_action(env, policy, last_observed_state)
        next_state = get_state_from_prices(env, next_price_set)

        if next_state ∈ visited_states
            break
        else
            push!(visited_states, next_state)
        end
    end

    profit_vects = get_profit_from_state.((env,), visited_states)

    profit_table = hcat(profit_vects...)'

    mean(profit_table, dims = 1)
end

function extract_sim_results(exp_list::Vector{AIAPCSummary})
    α_result = [ex.α for ex in exp_list if !(ex isa Exception)]
    β_result = [ex.β for ex in exp_list if !(ex isa Exception)]
    iterations_until_convergence =
        [ex.iterations_until_convergence[1] for ex in exp_list if !(ex isa Exception)]

    avg_profit_result =
        [mean(ex.convergence_profit) for ex in exp_list if !(ex isa Exception)]
    is_converged = [ex.is_converged for ex in exp_list if !(ex isa Exception)]
    df = DataFrame(
        α = α_result,
        β = β_result,
        π_bar = avg_profit_result,
        iterations_until_convergence = iterations_until_convergence,
        is_converged = is_converged,
    )
    return df
end
