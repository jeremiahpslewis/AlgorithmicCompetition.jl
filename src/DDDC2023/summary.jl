using Chain
using ReinforcementLearningCore, ReinforcementLearningBase
using DataFrames

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
    iterations_until_convergence::Vector{Int64}
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

    is_converged = Bool[]

    convergence_profit = get_convergence_profit_from_hook(hook)

    for i in (Symbol(1), Symbol(2))
        push!(is_converged, hook[i][1].is_converged)
    end

    return DDDCSummary(
        env.α,
        env.β,
        is_converged,
        env.data_demand_digital_params,
        convergence_profit,
        iterations_until_convergence,
    )
end

"""
    get_convergence_profit_from_env(env::DDDCEnv, policy::MultiAgentPolicy)

Returns the average profit of the agent, after convergence, over the convergence state or states (in the case of a cycle).
"""
function get_convergence_profit_from_hook(hook::AbstractHook)
    [mean(hook[p][2].rewards[101:end]) for p in [Symbol(1), Symbol(2)]]
end

"""
    extract_sim_results(exp_list::Vector{DDDCSummary})

Extracts the results of a simulation experiment, given a list of DDDCSummary objects, returns a `DataFrame`.
"""
function extract_sim_results(exp_list::Vector{DDDCSummary})
    α_result = [ex.α for ex in exp_list if !(ex isa Exception)]
    β_result = [ex.β for ex in exp_list if !(ex isa Exception)]
    iterations_until_convergence =
        [ex.iterations_until_convergence[1] for ex in exp_list if !(ex isa Exception)]

    avg_profit_result =
        [mean(ex.convergence_profit) for ex in exp_list if !(ex isa Exception)]
    profit_max =
        [maximum(ex.convergence_profit) for ex in exp_list if !(ex isa Exception)]
    profit_min =
        [minimum(ex.convergence_profit) for ex in exp_list if !(ex isa Exception)]

    is_converged = [ex.is_converged for ex in exp_list if !(ex isa Exception)]
    low_signal_quality_level = [ex.data_demand_digital_params.low_signal_quality_level for ex in exp_list if !(ex isa Exception)]
    high_signal_quality_boost = [ex.data_demand_digital_params.high_signal_quality_boost for ex in exp_list if !(ex isa Exception)]
    signal_quality_is_high = [ex.data_demand_digital_params.signal_quality_is_high for ex in exp_list if !(ex isa Exception)]
    frequency_high_demand = [ex.data_demand_digital_params.frequency_high_demand for ex in exp_list if !(ex isa Exception)]


    df = DataFrame(
        α = α_result,
        β = β_result,
        π_bar = avg_profit_result,
        profit_min = profit_min,
        profit_max = profit_max,        
        iterations_until_convergence = iterations_until_convergence,
        is_converged = is_converged,
        low_signal_quality_level = low_signal_quality_level,
        high_signal_quality_boost = high_signal_quality_boost,
        signal_quality_is_high = signal_quality_is_high,
        frequency_high_demand = frequency_high_demand,
    )
    return df
end
