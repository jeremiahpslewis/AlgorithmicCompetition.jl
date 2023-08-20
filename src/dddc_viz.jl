using CairoMakie
using Chain
using DataFrameMacros
using AlgebraOfGraphics
using CSV
using DataFrames
using Statistics
using Test

using AlgorithmicCompetition:
    post_prob_high_low_given_signal,
    post_prob_high_low_given_both_signals,
    draw_price_diagnostic,
    CompetitionParameters,
    CompetitionSolution,
    DataDemandDigitalParams,
    DDDCHyperParameters,
    draw_price_diagnostic

folder_name = "v0.0.5_data"

df_ = DataFrame.(CSV.File.(readdir(folder_name, join=true)))
df_ = vcat(df_...)

df__ = @chain df_ begin
    @transform(
        :price_response_to_demand_signal_mse =
            eval(Meta.parse(:price_response_to_demand_signal_mse)),
        :convergence_profit_demand_high =
            eval(Meta.parse(:convergence_profit_demand_high)),
        :convergence_profit_demand_low =
            eval(Meta.parse(:convergence_profit_demand_low)),
        :profit_vect = eval(Meta.parse(:profit_vect)),
        :profit_gain = eval(Meta.parse(:profit_gain)),
        :profit_gain_demand_high = eval(Meta.parse(:profit_gain_demand_high)),
        :profit_gain_demand_low = eval(Meta.parse(:profit_gain_demand_low)),
        :signal_is_strong_vect = eval(Meta.parse(:signal_is_strong)),
        :percent_unexplored_states_vect = eval(Meta.parse(:percent_unexplored_states)),
    )
    @transform!(
        @subset((:frequency_high_demand == 1) & (:weak_signal_quality_level == 1)),
        :price_response_to_demand_signal_mse = missing
    )
end

df___ = @chain df__ begin
    @transform(:signal_is_weak_vect = :signal_is_strong_vect .!= 1)
    @transform(:profit_mean = mean(:profit_vect))
    @transform(:percent_unexplored_states = mean(:percent_unexplored_states_vect))
    @transform(
        :percent_unexplored_states_demand_low_weak_signal_player =
            (:signal_is_strong ∈ ("Bool[0, 0]", "Bool[1, 1]")) |
            (:frequency_high_demand == 1) ? missing :
            :percent_unexplored_states_vect[:signal_is_weak_vect][1],
    )
    @transform(
        :percent_unexplored_states_demand_low_strong_signal_player =
            (:signal_is_strong ∈ ("Bool[0, 0]", "Bool[1, 1]")) |
            (:frequency_high_demand == 1) ? missing :
            :percent_unexplored_states_vect[:signal_is_strong_vect][1],
    )
    @transform(
        :percent_unexplored_states_demand_high_weak_signal_player =
            (:signal_is_strong ∈ ("Bool[0, 0]", "Bool[1, 1]")) ? missing :
            :percent_unexplored_states_vect[:signal_is_weak_vect][1],
    )
    @transform(
        :percent_unexplored_states_demand_high_strong_signal_player =
            (:signal_is_strong ∈ ("Bool[0, 0]", "Bool[1, 1]")) ? missing :
            :percent_unexplored_states_vect[:signal_is_strong_vect][1],
    )
    @transform(
        :profit_gain_demand_low_weak_signal_player =
            (:signal_is_strong ∈ ("Bool[0, 0]", "Bool[1, 1]")) |
            (:frequency_high_demand == 1) ? missing :
            :profit_gain_demand_low[:signal_is_weak_vect][1],
    )
    @transform(
        :profit_gain_demand_low_strong_signal_player =
            (:signal_is_strong ∈ ("Bool[0, 0]", "Bool[1, 1]")) |
            (:frequency_high_demand == 1) ? missing :
            :profit_gain_demand_low[:signal_is_strong_vect][1],
    )
    @transform(
        :profit_gain_demand_high_weak_signal_player =
            (:signal_is_strong ∈ ("Bool[0, 0]", "Bool[1, 1]")) ? missing :
            :profit_gain_demand_high[:signal_is_weak_vect][1],
    )
    @transform(
        :profit_gain_demand_high_strong_signal_player =
            (:signal_is_strong ∈ ("Bool[0, 0]", "Bool[1, 1]")) ? missing :
            :profit_gain_demand_high[:signal_is_strong_vect][1],
    )

    @transform(
        :profit_gain_weak_signal_player =
            (:signal_is_strong ∈ ("Bool[0, 0]", "Bool[1, 1]")) ? missing :
            :profit_gain[:signal_is_weak_vect][1],
    )
    @transform(
        :profit_gain_strong_signal_player =
            (:signal_is_strong ∈ ("Bool[0, 0]", "Bool[1, 1]")) ? missing :
            :profit_gain[:signal_is_strong_vect][1],
    )
end

df = @chain df___ begin
    @transform(:signal_is_weak_vect = :signal_is_strong_vect .!= 1)
    @transform(
        :convergence_profit_demand_low_weak_signal_player =
            (:signal_is_strong ∈ ("Bool[0, 0]", "Bool[1, 1]")) |
            (:frequency_high_demand == 1) ? missing :
            :convergence_profit_demand_low[:signal_is_weak_vect][1],
    )
    @transform(
        :convergence_profit_demand_low_strong_signal_player =
            (:signal_is_strong ∈ ("Bool[0, 0]", "Bool[1, 1]")) |
            (:frequency_high_demand == 1) ? missing :
            :convergence_profit_demand_low[:signal_is_strong_vect][1],
    )
    @transform(
        :convergence_profit_demand_high_weak_signal_player =
            (:signal_is_strong ∈ ("Bool[0, 0]", "Bool[1, 1]")) ? missing :
            :convergence_profit_demand_high[:signal_is_weak_vect][1],
    )
    @transform(
        :convergence_profit_demand_high_strong_signal_player =
            (:signal_is_strong ∈ ("Bool[0, 0]", "Bool[1, 1]")) ? missing :
            :convergence_profit_demand_high[:signal_is_strong_vect][1],
    )

    # TODO: Uncomment for later versions...
    # @transform(
    #     :convergence_profit_weak_signal_player = (:signal_is_strong ∈ ("Bool[0, 0]", "Bool[1, 1]")) ? missing : :convergence_profit[:signal_is_weak_vect][1],
    # )
    # @transform(
    #     :convergence_profit_strong_signal_player = (:signal_is_strong ∈ ("Bool[0, 0]", "Bool[1, 1]")) ? missing : :convergence_profit[:signal_is_strong_vect][1],
    # )
end

# Basic correctness assurance tests...
@test mean(mean.(df[!, :signal_is_weak_vect] .+ df[!, :signal_is_strong_vect])) == 1

@chain df begin
    @subset(:signal_is_strong ∉ ("Bool[0, 0]", "Bool[1, 1]"))
    @transform(
        :profit_gain_sum_1 =
            (:profit_gain_weak_signal_player + :profit_gain_strong_signal_player),
        :profit_gain_sum_2 = sum(:profit_gain),
    )
    @transform(:profit_gain_check = :profit_gain_sum_1 != :profit_gain_sum_2)
    @combine(sum(:profit_gain_check))
    @test _[1, :profit_gain_check_sum] == 0
end

@chain df begin
    @subset(:signal_is_strong ∉ ("Bool[0, 0]", "Bool[1, 1]"))
    @transform(
        :profit_gain_sum_1 = (
            :profit_gain_demand_high_weak_signal_player +
            :profit_gain_demand_high_strong_signal_player
        ),
        :profit_gain_sum_2 = sum(:profit_gain_demand_high),
    )
    @transform(:profit_gain_check = :profit_gain_sum_1 != :profit_gain_sum_2)
    @combine(sum(:profit_gain_check))
    @test _[1, :profit_gain_check_sum] == 0
end

@chain df begin
    @subset(
        (:signal_is_strong ∉ ("Bool[0, 0]", "Bool[1, 1]")) & (:frequency_high_demand != 1)
    )
    @transform(
        :profit_gain_sum_1 = (
            :profit_gain_demand_low_weak_signal_player +
            :profit_gain_demand_low_strong_signal_player
        ),
        :profit_gain_sum_2 = sum(:profit_gain_demand_low),
    )
    @transform(:profit_gain_check = :profit_gain_sum_1 != :profit_gain_sum_2)
    @combine(sum(:profit_gain_check))
    @test _[1, :profit_gain_check_sum] == 0
end

plt1 = @chain df begin
    @transform(:signal_is_strong = string(:signal_is_strong))
    data(_) *
    mapping(
        :frequency_high_demand,
        :profit_mean,
        color=:signal_is_strong => nonnumeric,
        row=:signal_is_strong,
    ) *
    visual(Scatter)
end
f1 = draw(plt1)
# save("plots/plot_1.svg", f1)

df_summary = @chain df begin
    @transform!(@subset(:signal_is_strong == "Bool[0, 1]"), :signal_is_strong = "Bool[1, 0]",)
    @transform(
        :price_response_to_demand_signal_mse_mean =
            @passmissing minimum(:price_response_to_demand_signal_mse)
    )
    @transform(
        :profit_gain_max = maximum(:profit_gain),
        :profit_gain_demand_high_max = maximum(:profit_gain_demand_high),
        :profit_gain_demand_low_max = maximum(:profit_gain_demand_low),
        :profit_gain_min = minimum(:profit_gain),
        :profit_gain_demand_high_min = minimum(:profit_gain_demand_high),
        :profit_gain_demand_low_min = minimum(:profit_gain_demand_low),
    )
    @groupby(:signal_is_strong, :weak_signal_quality_level, :frequency_high_demand)
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
        :percent_unexplored_states = mean(:percent_unexplored_states),
        :percent_unexplored_states_demand_low_weak_signal_player =
            mean(:percent_unexplored_states_demand_low_weak_signal_player),
        :percent_unexplored_states_demand_low_strong_signal_player =
            mean(:percent_unexplored_states_demand_low_strong_signal_player),
        :percent_unexplored_states_demand_high_weak_signal_player =
            mean(:percent_unexplored_states_demand_high_weak_signal_player),
        :percent_unexplored_states_demand_high_strong_signal_player =
            mean(:percent_unexplored_states_demand_high_strong_signal_player),
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
        # :convergence_profit_weak_signal_player = mean(:convergence_profit_weak_signal_player),
        # :convergence_profit_strong_signal_player = mean(:convergence_profit_strong_signal_player),
        :price_response_to_demand_signal_mse =
            (@passmissing mean(:price_response_to_demand_signal_mse_mean)),
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


f11 = @chain df_post_prob begin
    data(_) *
    mapping(
        :pr_high_demand,
        :post_prob_high_given_both_signals_high,
        color=:pr_signal_true => nonnumeric,
    ) *
    visual(Scatter)
end |> draw
save("plots/plot_11.svg", f11)

plt2 = @chain df_summary begin
    stack(
        [:profit_min_mean, :profit_max_mean],
        variable_name=:profit_variable_name,
        value_name=:profit_value,
    )
    @subset(:signal_is_strong == "Bool[0, 0]")
    @sort(:frequency_high_demand)
    data(_) *
    mapping(
        :frequency_high_demand,
        :profit_value,
        color=:profit_variable_name => nonnumeric,
        layout=:weak_signal_quality_level => nonnumeric,
    ) *
    (visual(Scatter) + visual(Lines))
end
f2 = draw(
    plt2,
    legend=(position=:top, titleposition=:left, framevisible=true, padding=5),
)
save("plots/plot_2.svg", f2)

plt20 = @chain df_summary begin
    @subset(:signal_is_strong == "Bool[0, 0]")
    @sort(:frequency_high_demand)
    data(_) *
    mapping(
        :frequency_high_demand,
        :profit_mean,
        color=:weak_signal_quality_level => nonnumeric,
    ) *
    (visual(Scatter) + visual(Lines))
end
f20 = draw(
    plt20,
    legend=(position=:top, titleposition=:left, framevisible=true, padding=5),
)
save("plots/plot_20.svg", f20)

plt21 = @chain df_summary begin
    @subset(
        (:signal_is_strong == "Bool[0, 0]") &
        !ismissing(:price_response_to_demand_signal_mse)
    )
    @sort(:frequency_high_demand)
    data(_) *
    mapping(
        :frequency_high_demand,
        :price_response_to_demand_signal_mse,
        color=:weak_signal_quality_level => nonnumeric,
    ) *
    (visual(Scatter) + visual(Lines))
end
# NOTE: freq_high_demand == 1 intersect weak_signal_quality_level == 1 is excluded, as the low demand states are never explored, so the price response to demand signal is not defined
f21 = draw(
    plt21,
    legend=(position=:top, titleposition=:left, framevisible=true, padding=5),
)
save("plots/plot_21.svg", f21)

plt22 = @chain df_summary begin
    stack(
        [:convergence_profit_demand_high, :convergence_profit_demand_low],
        variable_name=:demand_level,
        value_name=:profit,
    )
    @subset(:signal_is_strong == "Bool[0, 0]")
    @sort(:frequency_high_demand)
    data(_) *
    mapping(
        :frequency_high_demand,
        :profit,
        color=:demand_level => nonnumeric,
        layout=:weak_signal_quality_level => nonnumeric,
    ) *
    (visual(Scatter) + visual(Lines))
end
# NOTE: freq_high_demand == 1 intersect weak_signal_quality_level == 1 is excluded, as the low demand states are never explored, so the price response to demand signal is not defined
f22 = draw(
    plt22,
    legend=(position=:top, titleposition=:left, framevisible=true, padding=5),
)
save("plots/plot_22.svg", f22)

plt221 = @chain df_summary begin
    stack(
        [:profit_gain_min, :profit_gain_max],
        variable_name=:min_max,
        value_name=:profit_gain,
    )
    @sort(:frequency_high_demand)
    data(_) *
    mapping(
        :frequency_high_demand,
        :profit_gain,
        color=:min_max => nonnumeric,
        # columns = :weak_signal_quality_level => nonnumeric,
        rows=:signal_is_strong => nonnumeric,
    ) *
    (visual(Scatter) + visual(Lines))
end
# NOTE: freq_high_demand == 1 intersect weak_signal_quality_level == 1 is excluded, as the low demand states are never explored, so the price response to demand signal is not defined
f221 = draw(
    plt221,
    legend=(position=:top, titleposition=:left, framevisible=true, padding=5),
)
save("plots/plot_221.svg", f22)

# TODO: version of plt22, but where profit is normalized against demand scenario!
plt23 = @chain df_summary begin
    stack(
        [
            :profit_gain_demand_high_min,
            :profit_gain_demand_low_min,
            :profit_gain_demand_high_max,
            :profit_gain_demand_low_max,
        ],
        variable_name=:profit_gain_type,
        value_name=:profit_gain,
    )
    @subset(:signal_is_strong == "Bool[0, 0]")
    @sort(:frequency_high_demand)
    @transform(
        :profit_gain_type =
            replace(:profit_gain_type, "profit_gain_" => "")
    )
    data(_) *
    mapping(
        :frequency_high_demand,
        :profit_gain,
        color=:profit_gain_type => nonnumeric,
        layout=:weak_signal_quality_level => nonnumeric,
    ) *
    (visual(Scatter) + visual(Lines))
end
# NOTE: freq_high_demand == 1 intersect weak_signal_quality_level == 1 is excluded, as the low demand states are never explored, so the price response to demand signal is not defined
f23 = draw(
    plt23,
    # legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
)
save("plots/plot_23.svg", f23)

plt24 = @chain df_summary begin
    stack(
        [
            :profit_gain_demand_high_weak_signal_player,
            :profit_gain_demand_low_weak_signal_player,
            :profit_gain_demand_high_strong_signal_player,
            :profit_gain_demand_low_strong_signal_player,
        ],
        variable_name=:profit_gain_type,
        value_name=:profit_gain,
    )
    @subset((:signal_is_strong == "Bool[1, 0]"))
    @sort(:frequency_high_demand)
    @transform(:demand_level = replace(:profit_gain_type, r"profit_gain_demand_([a-z]+)_.*" => s"\1"))
    @transform(:signal_type = replace(:profit_gain_type, r"profit_gain_demand_[a-z]+_([a-z_]+)_signal_player" => s"\1"))
    @transform(:weak_signal_quality_level = string("Weak Signal Strength: ", :weak_signal_quality_level))
    @subset((:frequency_high_demand < 1) | (:demand_level == "high"))
    data(_) *
    mapping(
        :frequency_high_demand,
        :profit_gain,
        marker=:demand_level => nonnumeric => "Demand Level",
        layout= :weak_signal_quality_level => nonnumeric,
        color=:signal_type => "Signal Strength",
    ) *
    (visual(Scatter) + visual(Lines))
end
# NOTE: freq_high_demand == 1 intersect weak_signal_quality_level == 1 is excluded, as the low demand states are never explored, so the price response to demand signal is not defined
f24 = draw(
    plt24,
    legend=(
        position=:top,
        titleposition=:left,
        framevisible=true,
        padding=5,
        titlesize=10,
        labelsize=10,
    ),
    axis=(
        # title = x -> string(x, "aaa"),
        # subtitle = "(Solid line is profit for symmetric prices, shaded region shows range based on price options)",
        xlabel="High Demand Frequency",
        ylabel="Profit Gain",
        xticks=0.5:0.1:1,
        yticks=0:0.2:1,
        aspect=1,
        limits=(0.5, 1.05, 0.0, 1.0)
    ),)
save("plots/plot_24.svg", f24)

plt25 = @chain df_summary begin
    stack(
        [
            :convergence_profit_demand_high_weak_signal_player,
            :convergence_profit_demand_low_weak_signal_player,
            :convergence_profit_demand_high_strong_signal_player,
            :convergence_profit_demand_low_strong_signal_player,
        ],
        variable_name=:convergence_profit_type,
        value_name=:convergence_profit,
    )
    @subset((:signal_is_strong == "Bool[1, 0]") & (:frequency_high_demand != 1))
    @sort(:frequency_high_demand)
    @transform(
        :convergence_profit_type =
            replace(:convergence_profit_type, "convergence_profit_" => "")
    )
    data(_) *
    mapping(
        :frequency_high_demand,
        :convergence_profit,
        color=:convergence_profit_type => nonnumeric,
        layout=:weak_signal_quality_level => nonnumeric,
    ) *
    (visual(Scatter) + visual(Lines))
end
# NOTE: freq_high_demand == 1 intersect weak_signal_quality_level == 1 is excluded, as the low demand states are never explored, so the price response to demand signal is not defined
f25 = draw(
    plt25,
    legend=(
        position=:top,
        titleposition=:left,
        framevisible=true,
        padding=5,
        titlesize=10,
        labelsize=10,
    ),
    # axis = (width = 100, height = 100),
)
save("plots/plot_25.svg", f25)

plt26 = @chain df_summary begin
    stack(
        [
            :percent_unexplored_states_demand_high_weak_signal_player,
            :percent_unexplored_states_demand_low_weak_signal_player,
            :percent_unexplored_states_demand_high_strong_signal_player,
            :percent_unexplored_states_demand_low_strong_signal_player,
        ],
        variable_name=:profit_gain_type,
        value_name=:profit_gain,
    )
    @subset((:signal_is_strong == "Bool[1, 0]") & (:frequency_high_demand != 1))
    @sort(:frequency_high_demand)
    @transform(
        :convergence_profit_type =
            replace(:convergence_profit_type, "convergence_profit_" => "")
    )
    data(_) *
    mapping(
        :frequency_high_demand,
        :profit_gain,
        color=:profit_gain_type => nonnumeric,
        layout=:weak_signal_quality_level => nonnumeric,
    ) *
    (visual(Scatter) + visual(Lines))
end
# NOTE: freq_high_demand == 1 intersect weak_signal_quality_level == 1 is excluded, as the low demand states are never explored, so the price response to demand signal is not defined
f26 = draw(
    plt26,
    legend=(
        position=:top,
        titleposition=:left,
        framevisible=true,
        padding=5,
        titlesize=10,
        labelsize=10,
    ),
)
save("plots/plot_26.svg", f24)

plt27 = @chain df_summary begin
    stack(
        [
            :profit_gain_demand_high_weak_signal_player,
            :profit_gain_demand_low_weak_signal_player,
            :profit_gain_demand_high_strong_signal_player,
            :profit_gain_demand_low_strong_signal_player,
        ],
        variable_name=:profit_gain_type,
        value_name=:profit_gain,
    )
    @subset((:signal_is_strong == "Bool[1, 0]") & (:frequency_high_demand != 1))
    @sort(:frequency_high_demand)
    @transform(
        :demand_level =
            replace(:profit_gain_type, r"profit_gain_demand_([a-z]+)_.*" => s"\1")
    )
    @transform(
        :signal_type = replace(
            :profit_gain_type,
            r"profit_gain_demand_[a-z]+_([a-z_]+)_signal_player" => s"\1",
        )
    )
    data(_) *
    mapping(
        :frequency_high_demand,
        :profit_gain,
        color=:weak_signal_quality_level => nonnumeric,
        col=:demand_level => nonnumeric,
        row=:signal_type => nonnumeric,
    ) *
    (visual(Scatter) + visual(Lines))
end
# NOTE: freq_high_demand == 1 intersect weak_signal_quality_level == 1 is excluded, as the low demand states are never explored, so the price response to demand signal is not defined
f27 = draw(
    plt27,
    legend=(
        position=:top,
        titleposition=:left,
        framevisible=true,
        padding=5,
        titlesize=10,
        labelsize=10,
    ),
)
save("plots/plot_27.svg", f24)

plt3 = @chain df_summary begin
    @sort(:frequency_high_demand)
    data(_) *
    mapping(
        :frequency_high_demand,
        :iterations_until_convergence_mean,
        color=:weak_signal_quality_level => nonnumeric,
        layout=:signal_is_strong => nonnumeric,
    ) *
    (visual(Scatter) + visual(Lines))
end
f3 = draw(plt3)
save("plots/plot_3.svg", f3)


# TODO: Make this profit for low signal agent
plt4 = @chain df_summary begin
    data(_) *
    mapping(
        :frequency_high_demand,
        :profit_min_mean,
        color=:signal_is_strong => nonnumeric,
    ) *
    (visual(Scatter) + linear())
end
f4 = draw(plt4)
save("plots/plot_4.svg", f4)

# TODO: Make this profit for high signal agent
plt5 = @chain df_summary begin
    data(_) *
    mapping(
        :frequency_high_demand,
        :profit_max_mean,
        color=:signal_is_strong => nonnumeric,
    ) *
    (visual(Scatter) + linear())
end
f5 = draw(plt5)
save("plots/plot_5.svg", f5)

α = Float64(0.125)
β = Float64(4e-1)
δ = 0.95
ξ = 0.1
δ = 0.95
n_prices = 15
max_iter = Int(1e6) # 1e8
price_index = 1:n_prices

competition_params_dict = Dict(
    :high => CompetitionParameters(0.25, -0.25, (2, 2), (1, 1)),
    :low => CompetitionParameters(0.25, 0.25, (2, 2), (1, 1)),
)

competition_solution_dict =
    Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

data_demand_digital_params = DataDemandDigitalParams(
    weak_signal_quality_level=0.99,
    strong_signal_quality_level=0.995,
    signal_is_strong=[true, false],
    frequency_high_demand=0.9,
)

hyperparams = DDDCHyperParameters(
    α,
    β,
    δ,
    max_iter,
    competition_solution_dict,
    data_demand_digital_params;
    convergence_threshold=Int(1e5)
)

plt = draw_price_diagnostic(hyperparams)
f6 = draw(
    plt,
    axis=(
        title="Profit Levels across Price Options",
        subtitle="(Solid line is profit for symmetric prices, shaded region shows range based on price options)",
        xlabel="Competitor's Price Choice",
    ),
)
save("plots/plot_6.svg", f6)

# TODO: Look into different levels of low signal quality, whether the 'drop-off' happens subtly or abruptly
