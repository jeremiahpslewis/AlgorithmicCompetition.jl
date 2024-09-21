using AlgorithmicCompetition: get_demand_level, get_demand_signals
using DataFrames
using Statistics
using AlgebraOfGraphics
using CairoMakie
using Chain
using DataFrameMacros

frequency_high_demand_vect = 0.0:0.01:1.0
signal_scenarios = [[0.6, 0.9], [0.6, 0.6], [0.9, 0.9]]
n_samples = 10000

df_raw = DataFrame(is_high_demand = Bool[], frequency_high_demand = Float64[])
for f in frequency_high_demand_vect
    for i = 1:n_samples
        push!(df_raw, (get_demand_level(f), f))
    end
end

df = DataFrame()

for signals in signal_scenarios
    df_temp = @chain df_raw begin
        @transform(
            :signals = signals,
            :demand_signals = get_demand_signals(
                :is_high_demand,
                [true, false],
                signals[1],
                signals[2],
            )
        )
    end
    append!(df, df_temp)
end






df = @chain df begin
    @transform(
        :identical = sum(:demand_signals) ∈ [0, 2],
        :identical_and_correct = all(:is_high_demand == :demand_signals)
    )
    @transform(
        :minority_demand_state =
            (:is_high_demand & (:frequency_high_demand > 0.5)) |
            (!:is_high_demand & (:frequency_high_demand <= 0.5))
    )
    # @subset(:minority_demand_state)
end







df_raw = DataFrame(frequency_high_demand = Float64[], signal_strength = Float64[])
for i = 0:0.005:1
    for j = 0.5:0.2:0.9
        push!(df_raw, (i, j))
    end
end

df = @chain df_raw begin
    @transform(
        :prob_high_signal =
            :signal_strength * :frequency_high_demand +
            (1 - :signal_strength) * (1 - :frequency_high_demand)
    )
    @transform(
        :prob_low_signal =
            :signal_strength * (1 - :frequency_high_demand) +
            (1 - :signal_strength) * :frequency_high_demand
    )
    @transform(
        :post_prob_high_demand__cond_high_signal =
            :signal_strength * :frequency_high_demand / :prob_high_signal
    )
    @transform(
        :post_prob_high_demand__cond_low_signal =
            :signal_strength * (1 - :frequency_high_demand) / :prob_low_signal
    )
    stack(
        [:post_prob_high_demand__cond_high_signal, :post_prob_high_demand__cond_low_signal],
        variable_name = :signal,
        value_name = :probability_high_demand,
    )
    @transform(:probability_low_demand = 1 - :probability_high_demand)
    @transform(
        :probability_combined = max(:probability_high_demand, :probability_low_demand)
    )
    @transform(:signal = replace(:signal, "post_prob_high_demand__cond_" => ""))
    @transform(
        :weighted_probability_combined =
            :probability_combined *
            ifelse(:signal == "high_signal", :prob_high_signal, :prob_low_signal)
    )
    @sort(:frequency_high_demand)
end

plt_a_1 = @chain df begin
    data(_)
    mapping(
        :frequency_high_demand,
        :probability_combined,
        color = :signal_strength => nonnumeric,
        row = :signal,
    ) * visual(Lines)
end
draw(plt_a_1)


plt_a = @chain df begin
    @groupby(:frequency_high_demand, :signal_strength)
    @combine(:weighted_probability_combined = sum(:weighted_probability_combined))
    data(_)
    mapping(
        :frequency_high_demand,
        :weighted_probability_combined,
        color = :signal_strength => nonnumeric,
        col = :signal_strength => nonnumeric,
    ) * visual(Lines)
end
draw(plt_a)
