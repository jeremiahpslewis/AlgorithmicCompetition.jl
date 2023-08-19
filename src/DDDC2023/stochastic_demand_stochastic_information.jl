# demand is either high or low
# state is determined by prices (known to agents) and demand state (known to agents, but unreliable signal)

# state * 2 for high / low (signal given to agents)

# env.frequency_high_demand = 0.5
# env.demand_level_is_high = true / false
# env.high_demand_signal = [true, false]
# env.strong_signal_quality_level = 0.3 # high signal quality -> additive deviation from coin flip zero information signal
# env.weak_signal_quality_level = 0.1 # low signal quality -> base deviation from coin flip zero information signal

using Flux

@kwdef struct DataDemandDigitalParams
    weak_signal_quality_level::Float64 = 0.5 # probability of true signal (0.5 is lowest possible vale)
    strong_signal_quality_level::Float64 = 1.0 # probability of true signal (0.5 is lowest possible vale)
    signal_quality_is_high::Vector{Bool} = [false, false] # true if signal quality is high
    frequency_high_demand::Float64 = 0.5 # probability of high demand for a given episode
end

function get_demand_level(frequency_high_demand::Float64)
    rand() < frequency_high_demand ? true : false
end

get_demand_level(d::DataDemandDigitalParams) = get_demand_level(d.frequency_high_demand)


function get_demand_signals(
    demand_level_is_high::Bool,
    signal_quality_is_high::Vector{Bool},
    weak_signal_quality_level::Float64,
    strong_signal_quality_level::Float64,
)
    true_signal_probability =
        (weak_signal_quality_level .* .!signal_quality_is_high) .+
        (strong_signal_quality_level .* signal_quality_is_high)

    # Probability of true signal is a function of true signal probability
    reveal_true_signal = rand(2) .< true_signal_probability

    # Observed signal is 'whether we are lying' times 'whether true demand signal is high'
    observed_signal_demand_level_is_high = reveal_true_signal .== demand_level_is_high

    return observed_signal_demand_level_is_high
end

function get_demand_signals(d::DataDemandDigitalParams, is_high_demand_episode::Bool)
    get_demand_signals(
        is_high_demand_episode,
        d.signal_quality_is_high,
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

    return [num_high / denom_high, num_low / denom_low]
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

    return [num_high / denom_high, num_low / denom_low]
end
