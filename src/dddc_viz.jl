using CairoMakie
using Chain
using DataFrameMacros
using AlgebraOfGraphics
using CSV
using DataFrames
using Statistics

file_name = "simulation_results_dddc_2023-08-13T16:50:50.057"
df_ = DataFrame(CSV.File(file_name * ".csv"))

df = @chain df_ begin
    @transform(
        :price_response_to_demand_signal_mse =
            eval(Meta.parse(:price_response_to_demand_signal_mse)),
        :convergence_profit_demand_high =
            eval(Meta.parse(:convergence_profit_demand_high)),
        :convergence_profit_demand_low =
            eval(Meta.parse(:convergence_profit_demand_low)),
    )
    @transform!(@subset((:frequency_high_demand == 1) & (:low_signal_quality_level == 1)), :price_response_to_demand_signal_mse = missing)
end

plt1 = @chain df begin
    @transform(:signal_quality_is_high = string(:signal_quality_is_high))
    data(_) *
    mapping(
        :frequency_high_demand,
        :π_bar,
        color = :signal_quality_is_high => nonnumeric,
        row = :signal_quality_is_high,
    ) *
    visual(Scatter)
end
draw(plt1)

df_summary = @chain df begin
    @transform(
        :price_response_to_demand_signal_mse_min =
            @passmissing minimum(:price_response_to_demand_signal_mse)
    )
    @transform(
        :price_response_to_demand_signal_mse_max =
            @passmissing maximum(:price_response_to_demand_signal_mse)
    )
    @groupby(:signal_quality_is_high, :low_signal_quality_level, :frequency_high_demand)
    @combine(
        mean(:π_bar),
        mean(:iterations_until_convergence),
        mean(:profit_min),
        mean(:profit_max),
        :price_response_to_demand_signal_mse_min_mean = (@passmissing mean(:price_response_to_demand_signal_mse_min)),
        :price_response_to_demand_signal_mse_max_mean = (@passmissing mean(:price_response_to_demand_signal_mse_max)),
        :convergence_profit_demand_high = mean(:convergence_profit_demand_high),
        :convergence_profit_demand_low = mean(:convergence_profit_demand_low),
    )
end

# Question is how existence of low state destabilizes the high state / overall collusion and to what extent...
# Question becomes 'given signal, estimated demand state prob, which opponent do I believe I am competing against?' the low demand believing opponent or the high demand one...
# in the case where own and opponents' signals are public, the high-high signal state yields the following probability curve over high state base frequency:

df_post_prob = DataFrame(
    vcat([
        (
            pr_high_demand,
            pr_signal_true,
            post_prob_high_low_given_signal(pr_high_demand, pr_signal_true)[1],
            post_prob_high_low_given_both_signals(pr_high_demand, pr_signal_true)[1],
        ) for pr_high_demand = 0.5:0.01:1 for pr_signal_true = 0.5:0.1:1
    ]),
) # squared to reflect high-high signals, for each opponent, independently
rename!(
    df_post_prob,
    [
        :pr_high_demand,
        :pr_signal_true,
        :post_prob_high_given_signal_high,
        :post_prob_high_given_both_signals_high,
    ],
)


@chain df_post_prob begin
    data(_) *
    mapping(
        :pr_high_demand,
        :post_prob_high_given_both_signals_high,
        color = :pr_signal_true => nonnumeric,
    ) *
    visual(Scatter)
end |> draw

plt2 = @chain df_summary begin
    stack(
        [:profit_min_mean, :profit_max_mean],
        variable_name = :profit_variable_name,
        value_name = :profit_value,
    )
    @subset(:signal_quality_is_high == "Bool[0, 0]")
    @sort(:frequency_high_demand)
    data(_) *
    mapping(
        :frequency_high_demand,
        :profit_value,
        color = :profit_variable_name => nonnumeric,
        layout = :low_signal_quality_level => nonnumeric,
    ) *
    (visual(Scatter) + visual(Lines))
end
draw(
    plt2,
    legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
)

plt21 = @chain df_summary begin
    stack(
        [
            :price_response_to_demand_signal_mse_min_mean,
            :price_response_to_demand_signal_mse_max_mean,
        ],
        variable_name = :price_response_variable_name,
        value_name = :price_response_value,
    )
    @subset((:signal_quality_is_high == "Bool[0, 0]") & !ismissing(:price_response_value))
    @sort(:frequency_high_demand)
    data(_) *
    mapping(
        :frequency_high_demand,
        :price_response_value,
        color = :price_response_variable_name => nonnumeric,
        layout = :low_signal_quality_level => nonnumeric,
    ) *
    (visual(Scatter) + visual(Lines))
end
# NOTE: freq_high_demand == 1 intersect low_signal_quality_level == 1 is excluded, as the low demand states are never explored, so the price response to demand signal is not defined
draw(
    plt21,
    legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
)

plt22 = @chain df_summary begin
    stack(
        [
            :convergence_profit_demand_high,
            :convergence_profit_demand_low,
        ],
        variable_name = :demand_level,
        value_name = :profit,
    )
    @subset(:signal_quality_is_high == "Bool[0, 0]")
    @sort(:frequency_high_demand)
    data(_) *
    mapping(
        :frequency_high_demand,
        :profit,
        color = :demand_level => nonnumeric,
        layout = :low_signal_quality_level => nonnumeric,
    ) *
    (visual(Scatter) + visual(Lines))
end
# NOTE: freq_high_demand == 1 intersect low_signal_quality_level == 1 is excluded, as the low demand states are never explored, so the price response to demand signal is not defined
draw(
    plt22,
    legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
)

# TODO: version of plt22, but where profit is normalized against demand scenario!
plt23 = @chain df_summary begin
    stack(
        [
            :convergence_profit_demand_high,
            :convergence_profit_demand_low,
        ],
        variable_name = :demand_level,
        value_name = :profit,
    )
    @subset(:signal_quality_is_high == "Bool[0, 0]")
    @sort(:frequency_high_demand)
    data(_) *
    mapping(
        :frequency_high_demand,
        :profit,
        color = :demand_level => nonnumeric,
        layout = :low_signal_quality_level => nonnumeric,
    ) *
    (visual(Scatter) + visual(Lines))
end
# NOTE: freq_high_demand == 1 intersect low_signal_quality_level == 1 is excluded, as the low demand states are never explored, so the price response to demand signal is not defined
draw(
    plt23,
    legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
)


plt3 = @chain df_summary begin
    @sort(:frequency_high_demand)
    data(_) *
    mapping(
        :frequency_high_demand,
        :iterations_until_convergence_mean,
        color = :low_signal_quality_level => nonnumeric,
        layout = :signal_quality_is_high => nonnumeric,
    ) *
    (visual(Scatter) + visual(Lines))
end
draw(plt3)


# TODO: Make this profit for low signal agent
plt4 = @chain df_summary begin
    data(_) *
    mapping(
        :frequency_high_demand,
        :profit_min_mean,
        color = :signal_quality_is_high => nonnumeric,
    ) *
    visual(Scatter)
end
draw(plt4)

# TODO: Make this profit for high signal agent
plt5 = @chain df_summary begin
    data(_) *
    mapping(
        :frequency_high_demand,
        :profit_max_mean,
        color = :signal_quality_is_high => nonnumeric,
    ) *
    visual(Scatter)
end
draw(plt5)


# TODO: Look into different levels of low signal quality, whether the 'drop-off' happens subtly or abruptly
