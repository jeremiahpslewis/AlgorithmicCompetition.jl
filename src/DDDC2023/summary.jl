using Chain
using ReinforcementLearning
using DataFrames
using Flux: mse
using DataFrameMacros

"""
    DDDCSummary(α, β, is_converged, data_demand_digital_params, convergence_profit, convergence_profit_demand_high, convergence_profit_demand_low, profit_gain, profit_gain_demand_high, profit_gain_demand_low, iterations_until_convergence, price_response_to_demand_signal_mse, percent_demand_high)

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
    profit_gain::Vector{Float64}
    profit_gain_demand_high::Vector{Float64}
    profit_gain_demand_low::Vector{Float64}
    iterations_until_convergence::Vector{Int64}
    price_response_to_demand_signal_mse::Vector{Float64}
    percent_demand_high::Float64
    percent_unexplored_states::Vector{Float64}
end

"""
    extract_profit_vars(env::DDDCEnv)

Returns the Nash equilibrium and monopoly optimal profits, based on prices stored in `env`.
"""
function extract_profit_vars(env::DDDCEnv)
    p_Bert_nash_equilibrium = env.p_Bert_nash_equilibrium
    p_monop_opt = env.p_monop_opt
    competition_params = env.competition_params_dict

    π_N = Dict(
        i => π(
            p_Bert_nash_equilibrium[i],
            p_Bert_nash_equilibrium[i],
            competition_params[i],
        )[1] for i in [:high, :low]
    )
    π_M = Dict(
        i => π(p_monop_opt[i], p_monop_opt[i], competition_params[i])[1] for
        i in [:high, :low]
    )
    return (π_N, π_M)
end

"""
extract_quantity_vars(env::DDDCEnv)

Returns the Nash equilibrium and monopoly optimal quantities, based on prices stored in `env`.
"""
function extract_quantity_vars(env::DDDCEnv)
    p_Bert_nash_equilibrium = env.p_Bert_nash_equilibrium
    p_monop_opt = env.p_monop_opt
    competition_params = env.competition_params_dict

    π_N = Dict(
        i => Q(
            p_Bert_nash_equilibrium[i],
            p_Bert_nash_equilibrium[i],
            competition_params[i],
        )[1] for i in [:high, :low]
    )
    π_M = Dict(
        i => Q(p_monop_opt[i], p_monop_opt[i], competition_params[i])[1] for
        i in [:high, :low]
    )
    return (π_N, π_M)
end


function economic_summary(env::DDDCEnv, policy::MultiAgentPolicy, hook::AbstractHook)
    convergence_threshold = env.convergence_threshold
    iterations_until_convergence = Int64[
        hook[player][1].iterations_until_convergence for player in [Player(1), Player(2)]
    ]

    percent_demand_high = mean(hook[Player(1)][2].demand_state_high_vect)

    is_converged = Bool[]
    percent_unexplored_states = Float64[]
    convergence_profit = get_convergence_profit_from_hook(hook)


    for player_ in (Player(1), Player(2))
        push!(is_converged, hook[player_][1].is_converged)
        push!(percent_unexplored_states, mean(hook[player_][1].best_response_vector .== 0))
    end

    price_vs_demand_signal_counterfactuals =
        extract_price_vs_demand_signal_counterfactuals(env, hook)

    return DDDCSummary(
        env.α,
        env.β,
        is_converged,
        env.data_demand_digital_params,
        convergence_profit[:all],
        convergence_profit[:high],
        convergence_profit[:low],
        get.(profit_gain.(convergence_profit[:all], (env,)), :weighted, ""),
        get.(profit_gain.(convergence_profit[:high], (env,)), :high, ""),
        get.(profit_gain.(convergence_profit[:low], (env,)), :low, ""),
        iterations_until_convergence,
        [e_[1] for e_ in price_vs_demand_signal_counterfactuals],
        percent_demand_high,
        percent_unexplored_states,
    )
end

"""
    get_convergence_profit_from_env(env::DDDCEnv, policy::MultiAgentPolicy)

Returns the average profit of the agent, after convergence, over the convergence state or states (in the case of a cycle). Also returns the average profit for the high and low demand states.
"""
function get_convergence_profit_from_hook(hook::AbstractHook)
    demand_high = hook[Player(1)][2].demand_state_high_vect
    return Dict(
        :all => [mean(hook[p][2].rewards[101:end]) for p in [Player(1), Player(2)]],
        :high => [
            mean(hook[p][2].rewards[101:end][demand_high[101:end]]) for
            p in [Player(1), Player(2)]
        ],
        :low => [
            mean(hook[p][2].rewards[101:end][.!demand_high[101:end]]) for
            p in [Player(1), Player(2)]
        ],
    )
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

    convergence_profit =
        [mean(ex.convergence_profit) for ex in exp_list if !(ex isa Exception)]

    convergence_profit_demand_high =
        [ex.convergence_profit_demand_high for ex in exp_list if !(ex isa Exception)]
    convergence_profit_demand_low =
        [ex.convergence_profit_demand_low for ex in exp_list if !(ex isa Exception)]

    profit_vect = [ex.convergence_profit for ex in exp_list if !(ex isa Exception)]
    profit_max = [maximum(ex.convergence_profit) for ex in exp_list if !(ex isa Exception)]
    profit_min = [minimum(ex.convergence_profit) for ex in exp_list if !(ex isa Exception)]

    profit_gain = [ex.profit_gain for ex in exp_list if !(ex isa Exception)]
    profit_gain_demand_high =
        [ex.profit_gain_demand_high for ex in exp_list if !(ex isa Exception)]
    profit_gain_demand_low =
        [ex.profit_gain_demand_low for ex in exp_list if !(ex isa Exception)]

    is_converged = [ex.is_converged for ex in exp_list if !(ex isa Exception)]
    weak_signal_quality_level = [
        ex.data_demand_digital_params.weak_signal_quality_level for
        ex in exp_list if !(ex isa Exception)
    ]
    strong_signal_quality_level = [
        ex.data_demand_digital_params.strong_signal_quality_level for
        ex in exp_list if !(ex isa Exception)
    ]
    signal_is_strong = [
        ex.data_demand_digital_params.signal_is_strong for
        ex in exp_list if !(ex isa Exception)
    ]
    frequency_high_demand = [
        ex.data_demand_digital_params.frequency_high_demand for
        ex in exp_list if !(ex isa Exception)
    ]

    price_response_to_demand_signal_mse =
        [ex.price_response_to_demand_signal_mse for ex in exp_list if !(ex isa Exception)]

    percent_unexplored_states =
        [ex.percent_unexplored_states for ex in exp_list if !(ex isa Exception)]

    df = DataFrame(
        α = α_result,
        β = β_result,
        profit_vect = profit_vect,
        profit_min = profit_min,
        profit_max = profit_max,
        profit_gain = profit_gain,
        profit_gain_demand_high = profit_gain_demand_high,
        profit_gain_demand_low = profit_gain_demand_low,
        convergence_profit = convergence_profit,
        convergence_profit_demand_high = convergence_profit_demand_high,
        convergence_profit_demand_low = convergence_profit_demand_low,
        iterations_until_convergence = iterations_until_convergence,
        is_converged = is_converged,
        weak_signal_quality_level = weak_signal_quality_level,
        strong_signal_quality_level = strong_signal_quality_level,
        signal_is_strong = signal_is_strong,
        frequency_high_demand = frequency_high_demand,
        price_response_to_demand_signal_mse = price_response_to_demand_signal_mse,
        percent_demand_high = percent_demand_high,
        percent_unexplored_states = percent_unexplored_states,
    )
    return df
end


function extract_price_vs_demand_signal_counterfactuals(env::DDDCEnv, hook::AbstractHook)
    price_vs_demand_signal_counterfactuals = [
        extract_price_vs_demand_signal_counterfactuals(
            hook[player_][1].best_response_vector,
            env.state_space_lookup,
            env.price_options,
            env.n_prices,
        ) for player_ in [Player(1), Player(2)]
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

    # NOTE: mse is calculated only over the states explored by the agent for both signal levels
    price_mse = @chain price_counterfactual_df begin
        @subset((:price_given_high_demand_signal != 0) & (:price_given_low_demand_signal != 0))
        @combine(
            :price_mse =
                mse(:price_given_high_demand_signal, :price_given_low_demand_signal)
        )
        _[1, :price_mse]
    end
    return price_mse, price_counterfactual_df
end

"""
    profit_gain(π_hat, env::AIAPCEnv)

Returns the profit gain of the agent based on the current policy.
"""
function profit_gain(π_hat, env::DDDCEnv)
    π_N, π_M = extract_profit_vars(env)
    profit_gain_ =
        Dict(i => (mean(π_hat) - π_N[i]) / (π_M[i] - π_N[i]) for i in [:high, :low])
    π_N_weighted =
        π_N[:high] * env.data_demand_digital_params.frequency_high_demand +
        π_N[:low] * (1 - env.data_demand_digital_params.frequency_high_demand)
    π_M_weighted =
        π_M[:high] * env.data_demand_digital_params.frequency_high_demand +
        π_M[:low] * (1 - env.data_demand_digital_params.frequency_high_demand)

    profit_gain_weighted = (mean(π_hat) - π_N_weighted) / (π_M_weighted - π_N_weighted)
    return Dict(
        :high => profit_gain_[:high],
        :low => profit_gain_[:low],
        :weighted => profit_gain_weighted,
    )
end

function expand_and_extract_dddc(df::DataFrame)
    df__ = @chain df begin
        @transform!(
            @subset((:frequency_high_demand == 1) & (:weak_signal_quality_level == 1)),
            :price_response_to_demand_signal_mse = missing
        )
    end
    
    df___ = @chain df__ begin
        @transform(:signal_is_weak = :signal_is_strong .!= 1)
        @transform(:profit_mean = mean(:profit_vect))
        @transform(:mean_percent_unexplored_states = mean(:percent_unexplored_states))
        @transform(
            :percent_unexplored_states_weak_signal_player =
                (:signal_is_strong ∈ ([0, 0], [1, 1])) ? missing :
                :percent_unexplored_states[:signal_is_weak][1],
        )
        @transform(
            :percent_unexplored_states_strong_signal_player =
                (:signal_is_strong ∈ ([0, 0], [1, 1])) ? missing :
                :percent_unexplored_states[:signal_is_strong][1],
        )
        @transform(
            :profit_gain_demand_low_weak_signal_player =
                (:signal_is_strong ∈ ([0, 0], [1, 1])) |
                (:frequency_high_demand == 1) ? missing :
                :profit_gain_demand_low[:signal_is_weak][1],
        )
        @transform(
            :profit_gain_demand_low_strong_signal_player =
                (:signal_is_strong ∈ ([0, 0], [1, 1])) |
                (:frequency_high_demand == 1) ? missing :
                :profit_gain_demand_low[:signal_is_strong][1],
        )
        @transform(
            :profit_gain_demand_high_weak_signal_player =
                (:signal_is_strong ∈ ([0, 0], [1, 1])) ? missing :
                :profit_gain_demand_high[:signal_is_weak][1],
        )
        @transform(
            :profit_gain_demand_high_strong_signal_player =
                (:signal_is_strong ∈ ([0, 0], [1, 1])) ? missing :
                :profit_gain_demand_high[:signal_is_strong][1],
        )
    
        @transform(
            :profit_gain_weak_signal_player =
                (:signal_is_strong ∈ ([0, 0], [1, 1])) ? missing :
                :profit_gain[:signal_is_weak][1],
        )
        @transform(
            :profit_gain_strong_signal_player =
                (:signal_is_strong ∈ ([0, 0], [1, 1])) ? missing :
                :profit_gain[:signal_is_strong][1],
        )
    end
    
    df = @chain df___ begin
        @transform(:signal_is_weak = :signal_is_strong .!= 1)
        @transform(
            :convergence_profit_demand_low_weak_signal_player =
                (:signal_is_strong ∈ ([0, 0], [1, 1])) |
                (:frequency_high_demand == 1) ? missing :
                :convergence_profit_demand_low[:signal_is_weak][1],
        )
        @transform(
            :convergence_profit_demand_low_strong_signal_player =
                (:signal_is_strong ∈ ([0, 0], [1, 1])) |
                (:frequency_high_demand == 1) ? missing :
                :convergence_profit_demand_low[:signal_is_strong][1],
        )
        @transform(
            :convergence_profit_demand_high_weak_signal_player =
                (:signal_is_strong ∈ ([0, 0], [1, 1])) ? missing :
                :convergence_profit_demand_high[:signal_is_weak][1],
        )
        @transform(
            :convergence_profit_demand_high_strong_signal_player =
                (:signal_is_strong ∈ ([0, 0], [1, 1])) ? missing :
                :convergence_profit_demand_high[:signal_is_strong][1],
        )
    
        @transform(
            :convergence_profit_weak_signal_player =
                (:signal_is_strong ∈ ([0, 0], [1, 1])) ? missing :
                :profit_vect[:signal_is_weak][1],
        )
        @transform(
            :convergence_profit_strong_signal_player =
                (:signal_is_strong ∈ ([0, 0], [1, 1])) ? missing :
                :profit_vect[:signal_is_strong][1],
        )
    end

    return df
end

function construct_df_summary_dddc(df::DataFrame)
    df_summary = @chain df begin
        @transform!(@subset(:signal_is_strong == [0, 1]), :signal_is_strong = [1, 0],)
        @transform(
            :price_response_to_demand_signal_mse_mean =
                @passmissing minimum(:price_response_to_demand_signal_mse)
        )
        @transform(
            :weak_signal_quality_level_str =
                string("Weak Signal Strength: ", :weak_signal_quality_level)
        )
        @transform(
            :profit_gain_max = maximum(:profit_gain),
            :profit_gain_demand_high_max = maximum(:profit_gain_demand_high),
            :profit_gain_demand_low_max = maximum(:profit_gain_demand_low),
            :profit_gain_min = minimum(:profit_gain),
            :profit_gain_demand_high_min = minimum(:profit_gain_demand_high),
            :profit_gain_demand_low_min = minimum(:profit_gain_demand_low),
            :convergence_profit_demand_high = mean(:convergence_profit_demand_high),
            :convergence_profit_demand_low = mean(:convergence_profit_demand_low),
        )
        @groupby(
            :signal_is_strong,
            :weak_signal_quality_level,
            :weak_signal_quality_level_str,
            :frequency_high_demand,
        )
        @combine(
            :profit_mean = mean(:profit_mean),
            mean(:iterations_until_convergence),
            mean(:profit_min),
            mean(:profit_max),
            :profit_gain_min = mean(:profit_gain_min),
            :profit_gain_max = mean(:profit_gain_max),
            :profit_gain_demand_high_weak_signal_player =
                mean(:profit_gain_demand_high_weak_signal_player),
            :profit_gain_demand_low_weak_signal_player =
                mean(:profit_gain_demand_low_weak_signal_player),
            :profit_gain_demand_high_strong_signal_player =
                mean(:profit_gain_demand_high_strong_signal_player),
            :profit_gain_demand_low_strong_signal_player =
                mean(:profit_gain_demand_low_strong_signal_player),
            :mean_percent_unexplored_states = mean(:mean_percent_unexplored_states),
            :percent_unexplored_states_weak_signal_player =
                mean(:percent_unexplored_states_weak_signal_player),
            :percent_unexplored_states_strong_signal_player =
                mean(:percent_unexplored_states_strong_signal_player),
            :profit_gain_weak_signal_player = mean(:profit_gain_weak_signal_player),
            :profit_gain_strong_signal_player = mean(:profit_gain_strong_signal_player),
            :profit_gain_demand_high_min = mean(:profit_gain_demand_high_min),
            :profit_gain_demand_low_min = mean(:profit_gain_demand_low_min),
            :profit_gain_demand_high_max = mean(:profit_gain_demand_high_max),
            :profit_gain_demand_low_max = mean(:profit_gain_demand_low_max),
            :convergence_profit_demand_high_weak_signal_player =
                mean(:convergence_profit_demand_high_weak_signal_player),
            :convergence_profit_demand_low_weak_signal_player =
                mean(:convergence_profit_demand_low_weak_signal_player),
            :convergence_profit_demand_high_strong_signal_player =
                mean(:convergence_profit_demand_high_strong_signal_player),
            :convergence_profit_demand_low_strong_signal_player =
                mean(:convergence_profit_demand_low_strong_signal_player),
            :convergence_profit_weak_signal_player =
                mean(:convergence_profit_weak_signal_player),
            :convergence_profit_strong_signal_player =
                mean(:convergence_profit_strong_signal_player),
            :price_response_to_demand_signal_mse =
                (@passmissing mean(:price_response_to_demand_signal_mse_mean)),
            :convergence_profit_demand_high = mean(:convergence_profit_demand_high),
            :convergence_profit_demand_low = mean(:convergence_profit_demand_low),
        )
    end
    return df
end
