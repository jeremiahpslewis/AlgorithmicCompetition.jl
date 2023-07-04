# demand is either high or low
# state is determined by prices (known to agents) and demand state (known to agents, but unreliable signal)

# state * 2 for high / low (signal given to agents)

# env.frequency_high_demand = 0.5
# env.demand_level_is_high = true / false
# env.high_demand_signal = [true, false]
# env.high_signal_quality_boost = 0.3 # high signal quality -> additive deviation from coin flip zero information signal
# env.low_signal_quality_level = 0.1 # low signal quality -> base deviation from coin flip zero information signal

using Flux

struct DataDemandDigitalParams
    low_signal_quality_level::Float64 # probability of true signal is 0.5 + low_signal_quality_level
    high_signal_quality_boost::Float64 # probability of true signal is 0.5 + low_signal_quality_level + high_signal_quality_boost
    signal_quality_is_high::Vector{Bool} # true if signal quality is high
    frequency_high_demand::Float64 # probability of high demand for a given episode
    high_demand_state::Vector{Bool} # [true] if demand is high for a given episode
    demand_mode::Symbol # :high, :low, :random

    function DataDemandDigitalParams(; demand_mode::Symbol=:high)
        new(0.0, 0.0, [false, false], 0.5, rand(Bool, 1), demand_mode)
    end
end

function get_demand_level(frequency_high_demand::Float64)
    rand() < frequency_high_demand ? true : false
end


function get_demand_signals(demand_level_is_high::Bool, signal_quality_is_high::Vector{Bool}, low_signal_quality_level::Float64, high_signal_quality_boost::Float64)
    true_signal_probability = 0.5 + low_signal_quality_level .+ high_signal_quality_boost .* signal_quality_is_high

    # Probability of true signal is a function of true signal probability
    reveal_true_signal = rand(2) .< true_signal_probability

    # Observed signal is 'whether we are lying' times 'whether true demand signal is high'
    observed_signal_demand_level_is_high = reveal_true_signal .== demand_level_is_high

    return observed_signal_demand_level_is_high
end
