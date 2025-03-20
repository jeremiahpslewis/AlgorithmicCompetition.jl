# demand is either high or low
# state is determined by prices (known to agents) and demand state (known to agents, but unreliable signal)

# state * 2 for high / low (signal given to agents)

# env.frequency_high_demand = 0.5
# env.demand_level_is_high = true / false
# env.high_demand_signal = [true, false]
# env.strong_signal_quality_level = 0.3 # high signal quality -> additive deviation from coin flip zero information signal
# env.weak_signal_quality_level = 0.1 # low signal quality -> base deviation from coin flip zero information signal

@kwdef struct DDDCExperimentalParams
    weak_signal_quality_level::Float64 = 0.5 # probability of true signal (0.5 is lowest possible vale)
    strong_signal_quality_level::Float64 = 1.0 # probability of true signal (0.5 is lowest possible vale)
    signal_is_strong::Vector{Bool} = [false, false] # true if signal quality is high
    frequency_high_demand::Float64 = 0.5 # probability of high demand for a given episode
    trembling_hand_frequency::Float64 = 0.0     # Probability of trembling hand state
end

function get_demand_level(frequency_high_demand::Float64)
    rand() < frequency_high_demand ? true : false
end

get_demand_level(d::DDDCExperimentalParams) = get_demand_level(d.frequency_high_demand)


function get_demand_signals(
    demand_level_is_high::Bool,
    signal_is_strong::Vector{Bool},
    weak_signal_quality_level::Float64,
    strong_signal_quality_level::Float64,
)
    # If both signal qualities are zero, then we return a CONSTANT true signal, this is equivalent to Calvano et al. (2019)'s stochastic demand, NO signal
    if weak_signal_quality_level == 0.0 && strong_signal_quality_level == 0.0
        return [true, true]
    end

    # If both signal qualities are -1, then we return a common sunspot signal, e.g. identical for both players, but independent of demand
    if weak_signal_quality_level == -1.0 && strong_signal_quality_level == -1.0
        sunspot_value = rand(Bool)
        return [sunspot_value, sunspot_value]
    end

    # Manually compute the true signal probability for each player
    local prob1 =
        weak_signal_quality_level * (signal_is_strong[1] ? 0.0 : 1.0) +
        strong_signal_quality_level * (signal_is_strong[1] ? 1.0 : 0.0)
    local prob2 =
        weak_signal_quality_level * (signal_is_strong[2] ? 0.0 : 1.0) +
        strong_signal_quality_level * (signal_is_strong[2] ? 1.0 : 0.0)
    local reveal1 = rand() < prob1
    local reveal2 = rand() < prob2
    local obs1 = (reveal1 == demand_level_is_high)
    local obs2 = (reveal2 == demand_level_is_high)
    return Bool[obs1, obs2]
end

function get_demand_signals(d::DDDCExperimentalParams, is_high_demand_episode::Bool)
    get_demand_signals(
        is_high_demand_episode,
        d.signal_is_strong,
        d.weak_signal_quality_level,
        d.strong_signal_quality_level,
    )
end

function post_prob_high_low_given_signal(pr_high_demand, pr_signal_true)
    denom_high =
        pr_high_demand * pr_signal_true + (1 - pr_high_demand) * (1 - pr_signal_true)
    denom_low =
        (1 - pr_high_demand) * pr_signal_true + pr_high_demand * (1 - pr_signal_true)

    num_high = pr_high_demand * pr_signal_true
    num_low = (1 - pr_high_demand) * pr_signal_true

    return (num_high / denom_high, num_low / denom_low)
end

function post_prob_high_low_given_both_signals(pr_high_demand, pr_signal_true)
    denom_high =
        pr_high_demand * pr_signal_true^2 +
        (1 - pr_high_demand) * (1 - pr_signal_true)^2 +
        2 * pr_high_demand * pr_signal_true * (1 - pr_signal_true)
    denom_low =
        (1 - pr_high_demand) * pr_signal_true^2 +
        pr_high_demand * (1 - pr_signal_true)^2 +
        2 * (1 - pr_high_demand) * pr_signal_true * (1 - pr_signal_true)

    num_high = pr_high_demand * pr_signal_true^2
    num_low = (1 - pr_high_demand) * pr_signal_true^2

    return (num_high / denom_high, num_low / denom_low)
end
