using Chain
using ReinforcementLearningCore, ReinforcementLearningBase
using DataFrames
using Flux: mse
"""
    DDDCSummary(α, β, is_converged, data_demand_digital_params, convergence_profit, iterations_until_convergence)

A struct to store the summary of an DDDC experiment.
"""
struct DDDCSummary
    α::Float64
    β::Float64
    is_converged::Vector{Bool}
    data_demand_digital_params::DataDemandDigitalParams
    convergence_profit::Vector{Float64}
    convergence_profit_demand_high::Vector{Float64}
    convergence_profit_demand_low::Vector{Float64}
    iterations_until_convergence::Vector{Int64}
    price_response_to_demand_signal_mse::Vector{Float64}
    percent_demand_high::Float64
end

"""
    extract_profit_vars(env::DDDCEnv)

Returns the Nash equilibrium and monopoly optimal profits, based on prices stored in env.
"""
function extract_profit_vars(env::DDDCEnv)
    # TODO: Profit extraction, factoring in the demand mode
    # p_Bert_nash_equilibrium = env.p_Bert_nash_equilibrium
    # p_monop_opt = env.p_monop_opt
    # competition_params = env.competition_params_dict[:high]

    # π_N = π(p_Bert_nash_equilibrium, p_Bert_nash_equilibrium, competition_params)[1]
    # π_M = π(p_monop_opt, p_monop_opt, competition_params)[1]
    # return (π_N, π_M)
end

function economic_summary(env::DDDCEnv, policy::MultiAgentPolicy, hook::AbstractHook)
    convergence_threshold = env.convergence_threshold
    iterations_until_convergence = Int64[
        hook[player][1].iterations_until_convergence for player in [Symbol(1), Symbol(2)]
    ]

    percent_demand_high = mean(hook[Symbol(1)][2].demand_state_high_vect)

    is_converged = Bool[]

    convergence_profit, convergence_profit_demand_high, convergence_profit_demand_low = get_convergence_profit_from_hook(hook)

    for i in (Symbol(1), Symbol(2))
        push!(is_converged, hook[i][1].is_converged)
    end

    price_vs_demand_signal_counterfactuals =
        extract_price_vs_demand_signal_counterfactuals(env, hook)

    return DDDCSummary(
        env.α,
        env.β,
        is_converged,
        env.data_demand_digital_params,
        convergence_profit,
        convergence_profit_demand_high,
        convergence_profit_demand_low,
        iterations_until_convergence,
        [e_[1] for e_ in price_vs_demand_signal_counterfactuals],
        percent_demand_high,
    )
end

"""
    get_convergence_profit_from_env(env::DDDCEnv, policy::MultiAgentPolicy)

Returns the average profit of the agent, after convergence, over the convergence state or states (in the case of a cycle). Also returns the average profit for the high and low demand states.
"""
function get_convergence_profit_from_hook(hook::AbstractHook)
    demand_high = hook[Symbol(1)][2].demand_state_high_vect
    return mean(hook[p][2].rewards[101:end]),
    [sum(hook[p][2].rewards[101:end] .* demand_high[101:end]) / sum(demand_high[101:end]) for p in [Symbol(1), Symbol(2)]],
    [sum(hook[p][2].rewards[101:end] .* .! demand_high[101:end]) / sum(.! demand_high[101:end]) for p in [Symbol(1), Symbol(2)]]
end

"""
    extract_sim_results(exp_list::Vector{DDDCSummary})

Extracts the results of a simulation experiment, given a list of DDDCSummary objects, returns a `DataFrame`.
"""
function extract_sim_results(exp_list::Vector{DDDCSummary})
    α_result = [ex.α for ex in exp_list if !(ex isa Exception)]
    β_result = [ex.β for ex in exp_list if !(ex isa Exception)]
    percent_demand_high = [ex.percent_demand_high for ex in exp_list if !(ex isa Exception)]
    iterations_until_convergence =
        [ex.iterations_until_convergence[1] for ex in exp_list if !(ex isa Exception)]

    avg_profit_result =
        [mean(ex.convergence_profit) for ex in exp_list if !(ex isa Exception)]

    convergence_profit_demand_high =
        [mean(ex.convergence_profit_demand_high) for ex in exp_list if !(ex isa Exception)]
    convergence_profit_demand_low =
        [mean(ex.convergence_profit_demand_low) for ex in exp_list if !(ex isa Exception)]

    profit_vect = [ex.convergence_profit for ex in exp_list if !(ex isa Exception)]
    profit_max = [maximum(ex.convergence_profit) for ex in exp_list if !(ex isa Exception)]
    profit_min = [minimum(ex.convergence_profit) for ex in exp_list if !(ex isa Exception)]

    is_converged = [ex.is_converged for ex in exp_list if !(ex isa Exception)]
    low_signal_quality_level = [
        ex.data_demand_digital_params.low_signal_quality_level for
        ex in exp_list if !(ex isa Exception)
    ]
    high_signal_quality_level = [
        ex.data_demand_digital_params.high_signal_quality_level for
        ex in exp_list if !(ex isa Exception)
    ]
    signal_quality_is_high = [
        ex.data_demand_digital_params.signal_quality_is_high for
        ex in exp_list if !(ex isa Exception)
    ]
    frequency_high_demand = [
        ex.data_demand_digital_params.frequency_high_demand for
        ex in exp_list if !(ex isa Exception)
    ]

    price_response_to_demand_signal_mse =
        [ex.price_response_to_demand_signal_mse for ex in exp_list if !(ex isa Exception)]

    df = DataFrame(
        α = α_result,
        β = β_result,
        π_bar = avg_profit_result,
        profit_vect = profit_vect,
        profit_min = profit_min,
        profit_max = profit_max,
        convergence_profit_demand_high = convergence_profit_demand_high,
        convergence_profit_demand_low = convergence_profit_demand_low,
        iterations_until_convergence = iterations_until_convergence,
        is_converged = is_converged,
        low_signal_quality_level = low_signal_quality_level,
        high_signal_quality_level = high_signal_quality_level,
        signal_quality_is_high = signal_quality_is_high,
        frequency_high_demand = frequency_high_demand,
        price_response_to_demand_signal_mse = price_response_to_demand_signal_mse,
        percent_demand_high = percent_demand_high,
    )
    return df
end


function extract_price_vs_demand_signal_counterfactuals(env::DDDCEnv, hook::AbstractHook)
    best_response_vector = hook[Symbol(1)][1].best_response_vector
    price_vs_demand_signal_counterfactuals = [
        extract_price_vs_demand_signal_counterfactuals(
            hook[player_][1].best_response_vector,
            env.state_space_lookup,
            env.price_options,
            env.n_prices,
        ) for player_ in [Symbol(1), Symbol(2)]
    ]
    return price_vs_demand_signal_counterfactuals
end

function extract_price_vs_demand_signal_counterfactuals(
    best_response_vector,
    state_space_lookup,
    price_options,
    n_prices,
)
    price_counterfactual_vect = []

    for i = 1:n_prices
        for j = 1:n_prices
            for k = 1:2
                # The price that a player would choose if given signal 1 and signal 2 (e.g. high=1 or low=2), conditional on memory (prices and previous signals)
                best_response_price_indices =
                    best_response_vector[state_space_lookup[i, j, :, k]]
                if all(best_response_price_indices .> 0)
                    price_counterfactuals = price_options[best_response_price_indices]
                else
                    price_counterfactuals = [0, 0]
                end
                push!(price_counterfactual_vect, ((i, j, k), price_counterfactuals...))
            end
        end
    end

    price_counterfactual_df = DataFrame(
        price_counterfactual_vect,
        [:memory_index, :price_given_high_demand_signal, :price_given_low_demand_signal],
    )

    price_mse = mse(
        price_counterfactual_df[!, :price_given_high_demand_signal],
        price_counterfactual_df[!, :price_given_low_demand_signal],
    )

    return price_mse, price_counterfactual_df
end
