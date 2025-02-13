using CairoMakie
using Chain
using DataFrameMacros
using AlgebraOfGraphics
using Statistics
using Test
using DataFrames

using AlgorithmicCompetition:
    post_prob_high_low_given_signal,
    post_prob_high_low_given_both_signals,
    CompetitionParameters,
    CompetitionSolution,
    DataDemandDigitalParams,
    DDDCHyperParameters
using AlgorithmicCompetition
using Arrow

include("viz/price_diagnostics.jl")

rebuild_summary_files = false
rebuild_overall_summary = true
df_summary_arrow_cache_path = "data_final/dddc_v0.0.9_data_summary.arrow"

arrow_folders = filter!(
    x -> occursin(r"dddc_version=2025-01-14-dddc-mini", x),
    readdir("data", join = true),
)
arrow_files = vcat(
    [filter(y -> occursin(".arrow", y), readdir(x, join = true)) for x in arrow_folders]...,
)

# TODO: For decile demand-frequency binned data, round / group to avoid losing data...
# TODO: revert to df_summary once all summary files are tested...
is_summary_file = occursin.(("df_summary",), arrow_files)
df_summary_ = arrow_files[is_summary_file]
df_raw_ = arrow_files[.!is_summary_file]

if rebuild_summary_files
    rm.(df_summary_)
    for i in eachindex(df_raw_)
        df = DataFrame(Arrow.Table(df_raw_[i]))
        if nrow(df) > 0
            df = AlgorithmicCompetition.expand_and_extract_dddc(df)
            df_summary = AlgorithmicCompetition.construct_df_summary_dddc(df)
            Arrow.write(
                replace(df_raw_[i], ".arrow" => "_df_summary_rebuilt.arrow"),
                df_summary,
            )
        end
    end
end

if rebuild_overall_summary
    arrows_ = DataFrame.(Arrow.Table.(df_summary_))

    for i in eachindex(arrows_)
        arrows_[i][!, "metadata"] .= df_summary_[i]
        if "signal_is_strong" in names(arrows_[i])
            arrows_[i] = select!(arrows_[i], Not(:signal_is_strong))
        end
    end

    df_summary = vcat(arrows_...)
    df_summary = AlgorithmicCompetition.reduce_dddc(df_summary)#, round_parameters=false)
    mkpath("data_final")
    Arrow.write(df_summary_arrow_cache_path, df_summary)
end
# mkpath("data_final")
# arrow_file_name = "data_final/dddc_v0.0.8_data.arrow"
# Arrow.write(arrow_file_name, df_)

mkpath("plots/dddc")
# df_ = DataFrame(Arrow.read(arrow_file_name))

# n_simulations_dddc = @chain df_ @subset(
#     (:weak_signal_quality_level == 1) &
#     (:frequency_high_demand == 1) &
#     (:strong_signal_quality_level == :weak_signal_quality_level)
# ) nrow()

# @test (101 * 101 * 2 * n_simulations_dddc) == nrow(df_)

# df = AlgorithmicCompetition.expand_and_extract_dddc(df_)

# Basic correctness assurance tests...
# @test mean(mean.(df[!, :signal_is_weak] .+ df[!, :signal_is_strong])) == 1

# @chain df begin
#     @subset(:signal_is_strong ∉ ([0, 0], [1, 1]))
#     @transform(
#         :profit_gain_sum_1 =
#             (:profit_gain_weak_signal_player + :profit_gain_strong_signal_player),
#         :profit_gain_sum_2 = sum(:profit_gain),
#     )
#     @transform(:profit_gain_check = :profit_gain_sum_1 != :profit_gain_sum_2)
#     @combine(sum(:profit_gain_check))
#     @test _[1, :profit_gain_check_sum] == 0
# end

# @chain df begin
#     @subset(:signal_is_strong ∉ ([0, 0], [1, 1]))
#     @transform(
#         :profit_gain_sum_1 = (
#             :profit_gain_demand_high_weak_signal_player +
#             :profit_gain_demand_high_strong_signal_player
#         ),
#         :profit_gain_sum_2 = sum(:profit_gain_demand_high),
#     )
#     @transform(:profit_gain_check = :profit_gain_sum_1 != :profit_gain_sum_2)
#     @combine(sum(:profit_gain_check))
#     @test _[1, :profit_gain_check_sum] == 0
# end

# @chain df begin
#     @subset(
#         (:signal_is_strong ∉ ([0, 0], [1, 1])) & (:frequency_high_demand != 1)
#     )
#     @transform(
#         :profit_gain_sum_1 = (
#             :profit_gain_demand_low_weak_signal_player +
#             :profit_gain_demand_low_strong_signal_player
#         ),
#         :profit_gain_sum_2 = sum(:profit_gain_demand_low),
#     )
#     @transform(:profit_gain_check = :profit_gain_sum_1 != :profit_gain_sum_2)
#     @combine(sum(:profit_gain_check))
#     @test _[1, :profit_gain_check_sum] == 0
# end

# plt1 = @chain df begin
#     @transform(:signal_is_strong = string(:signal_is_strong))
#     data(_) *
#     mapping(
#         :frequency_high_demand => "High Demand Frequency",
#         :profit_mean => "Average Profit",
#         color = :signal_is_strong => nonnumeric,
#         row = :signal_is_strong,
#     ) *
#     visual(Scatter)
# end
# f1 = draw(plt1)
# save("plots/dddc/plot_1.svg", f1)

# df_summary = AlgorithmicCompetition.construct_df_summary_dddc(df_)
# @assert nrow(df_summary) == 20402
# TODO: Rereduce summary data across all runs!

# Question is how existence of low state destabilizes the high state / overall collusion and to what extent...
# Question becomes 'given signal, estimated demand state prob, which opponent do I believe I am competing against?' the low demand believing opponent or the high demand one...
# in the case where own and opponents' signals are public, the high-high signal state yields the following probability curve over high state base frequency:

strong_signal_level = 0.9
df_summary = AlgorithmicCompetition.reduce_dddc(DataFrame(Arrow.Table(df_summary_arrow_cache_path)))

df_post_prob = DataFrame(
    vcat([
        (
            pr_high_demand,
            pr_signal_true,
            post_prob_high_low_given_signal(pr_high_demand, pr_signal_true)[1],
            post_prob_high_low_given_both_signals(pr_high_demand, pr_signal_true)[1],
            pr_high_demand^2 * pr_signal_true,
        ) for pr_high_demand = 0.0:0.01:1 for pr_signal_true = 0.5:0.1:1
    ]),
) # squared to reflect high-high signals, for each opponent, independently
rename!(
    df_post_prob,
    [
        :pr_high_demand,
        :pr_signal_true,
        :post_prob_high_given_signal_high,
        :post_prob_high_given_both_signals_high,
        :state_and_signals_agree_prob,
    ],
)


f11 = @chain df_post_prob begin
    data(_) *
    mapping(
        :pr_high_demand,
        :state_and_signals_agree_prob,
        color = :pr_signal_true => nonnumeric => "Signal Strength",
    ) *
    visual(Scatter)
    draw(
        axis = (
            xticks = 0.0:0.1:1,
            yticks = 0:0.1:1,
            xlabel = "Probability High Demand",
            ylabel = "Probability High Demand and Opponent Signal High Given Own Signal High",
        ),
    )
end
save("plots/dddc/plot_11.svg", f11)

plt2 = @chain df_summary begin
    stack(
        [:profit_min, :profit_max],
        variable_name = :profit_variable_name,
        value_name = :profit_value,
    )
    @subset(
        (:strong_signal_quality_level == :weak_signal_quality_level) &
        (:weak_signal_quality_level == round(:weak_signal_quality_level; digits = 1))
    )
    @sort(:frequency_high_demand)
    @transform(
        :weak_signal_quality_level_str =
            string("Signal Strength: ", :weak_signal_quality_level),
        :profit_variable_name = replace(:profit_variable_name, "_" => " "),
    )
    data(_) *
    mapping(
        :frequency_high_demand => "High Demand Frequency",
        :profit_value => "Average Profit",
        color = :profit_variable_name => nonnumeric => "",
        layout = :weak_signal_quality_level_str => nonnumeric,
    ) *
    (visual(Lines))
end
f2 = draw(
    plt2,
    legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
)
save("plots/dddc/plot_2.svg", f2)

plt20 = @chain df_summary begin
    @subset(
        (:strong_signal_quality_level == :weak_signal_quality_level) &
        (:weak_signal_quality_level == round(:weak_signal_quality_level; digits = 1))
    )
    @sort(:frequency_high_demand)
    data(_) *
    mapping(
        :frequency_high_demand => "High Demand Frequency",
        :profit_mean => "Average Profit",
        color = :weak_signal_quality_level => nonnumeric => "Symmetric Signal Strength",
    ) *
    visual(Lines)
end
f20 = draw(
    plt20,
    legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
)
save("plots/dddc/plot_20.svg", f20)

plt201 = @chain df_summary begin
    @subset(:strong_signal_quality_level == :weak_signal_quality_level)
    data(_) *
    mapping(
        :frequency_high_demand => "High Demand Frequency",
        :weak_signal_quality_level => "Symmetric Signal Strength",
        :profit_mean => "Average Profit",
    ) *
    visual(Heatmap)
end
f201 = draw(
    plt201,
    legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
)
save("plots/dddc/plot_201.svg", f201)


plt21 = @chain df_summary begin
    @subset(
        (:strong_signal_quality_level == :weak_signal_quality_level) &
        !ismissing(:price_response_to_demand_signal_mse) &
        (:weak_signal_quality_level == round(:weak_signal_quality_level; digits = 1))
    )
    @sort(:frequency_high_demand)
    data(_) *
    mapping(
        :frequency_high_demand => "High Demand Frequency",
        :price_response_to_demand_signal_mse => "Mean Squared Error Price Difference by Demand Signal",
        color = :weak_signal_quality_level => nonnumeric => "Symmetric Signal Strength",
    ) *
    visual(Lines)
end
# NOTE: freq_high_demand == 1 intersect weak_signal_quality_level == 1 is excluded, as the low demand states are never explored, so the price response to demand signal is not defined
f21 = draw(
    plt21,
    legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
)
save("plots/dddc/plot_21.svg", f21)


plt211 = @chain df_summary begin
    @subset(
        (:strong_signal_quality_level == :weak_signal_quality_level) &
        !ismissing(:price_response_to_demand_signal_mse)
    )

    data(_) *
    mapping(
        :frequency_high_demand => "High Demand Frequency",
        :weak_signal_quality_level => "Symmetric Signal Strength",
        :price_response_to_demand_signal_mse => "Mean Squared Error Price Difference by Demand Signal",
    ) *
    visual(Heatmap)
end
f211 = draw(
    plt211,
    legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
)
save("plots/dddc/plot_211.svg", f211)

plt22 = @chain df_summary begin
    stack(
        [:convergence_profit_demand_high, :convergence_profit_demand_low],
        variable_name = :demand_level,
        value_name = :profit,
    )
    @subset(
        (:strong_signal_quality_level == :weak_signal_quality_level) &
        (:weak_signal_quality_level == round(:weak_signal_quality_level; digits = 1))
    )
    @subset(!ismissing(:profit))
    @transform(:demand_level = replace(:demand_level, "convergence_profit_demand_" => ""))
    @sort(:frequency_high_demand)
    @transform(
        :weak_signal_quality_level_str =
            string("Signal Strength: ", :weak_signal_quality_level)
    )
    data(_) *
    mapping(
        :frequency_high_demand => "High Demand Frequency",
        :profit => "Average Profit",
        color = :demand_level => nonnumeric => "Demand Level",
        layout = :weak_signal_quality_level_str => nonnumeric,
    ) *
    visual(Lines)
end
# NOTE: freq_high_demand == 1 intersect weak_signal_quality_level == 1 is excluded, as the low demand states are never explored, so the price response to demand signal is not defined
f22 = draw(
    plt22,
    legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
)
save("plots/dddc/plot_22.svg", f22)

plt222 = @chain df_summary begin
    stack(
        [:convergence_profit_demand_high, :convergence_profit_demand_low],
        variable_name = :demand_level,
        value_name = :profit,
    )
    @subset(:strong_signal_quality_level == :weak_signal_quality_level)
    @transform(
        :demand_level =
            replace(:demand_level, "convergence_profit_demand_" => "Demand: ")
    )
    data(_) *
    mapping(
        :frequency_high_demand => "High Demand Frequency",
        :weak_signal_quality_level => "Symmetric Signal Strength",
        :profit => "Average Profit",
        layout = :demand_level => nonnumeric => "Demand Level",
    ) *
    visual(Heatmap)
end
f222 = draw(
    plt222,
    # legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
)
save("plots/dddc/plot_222.svg", f222)

plt221 = @chain df_summary begin
    @subset(
        (:strong_signal_quality_level == :weak_signal_quality_level) &
        (:weak_signal_quality_level == round(:weak_signal_quality_level; digits = 1))
    )
    stack(
        [:profit_gain_min, :profit_gain_max],
        variable_name = :min_max,
        value_name = :profit_gain,
    )
    @transform(
        :min_max = (:min_max == "profit_gain_min" ? "Per-Trial Min" : "Per-Trial Max")
    )
    @sort(:frequency_high_demand)
    @transform(
        :weak_signal_quality_level_str =
            string("Signal Strength: ", :weak_signal_quality_level)
    )
    data(_) *
    mapping(
        :frequency_high_demand => "High Demand Frequency",
        :profit_gain,
        color = :min_max => nonnumeric => "",
        # columns = :weak_signal_quality_level => nonnumeric,
        layout = :weak_signal_quality_level_str => nonnumeric,
    ) *
    visual(Lines)
end
# NOTE: freq_high_demand == 1 intersect weak_signal_quality_level == 1 is excluded, as the low demand states are never explored, so the price response to demand signal is not defined
f221 = draw(
    plt221,
    legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
    axis = (xlabel = "High Demand Frequency", ylabel = "Profit Gain"),
)
save("plots/dddc/plot_221.svg", f221)

plt223 = @chain df_summary begin
    @subset(:strong_signal_quality_level == :weak_signal_quality_level)
    stack(
        [:profit_gain_min, :profit_gain_max],
        variable_name = :min_max,
        value_name = :profit_gain,
    )
    @transform(
        :min_max = (:min_max == "profit_gain_min" ? "Per-Trial Min" : "Per-Trial Max")
    )
    data(_) *
    mapping(
        :frequency_high_demand => "High Demand Frequency",
        :weak_signal_quality_level => "Symmetric Signal Strength",
        :profit_gain => "Profit Gain",
        layout = :min_max => nonnumeric => "",
    ) *
    visual(Heatmap)
end
f223 = draw(
    plt223,
    axis = (xticks = 0.0:0.2:1,),
    legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
)
save("plots/dddc/plot_223.svg", f223)

plt23 = @chain df_summary begin
    @subset(
        (:strong_signal_quality_level == :weak_signal_quality_level) &
        (:weak_signal_quality_level == round(:weak_signal_quality_level; digits = 1))
    )
    @transform(
        :profit_gain_demand_all_min = :profit_gain_min,
        :profit_gain_demand_all_max = :profit_gain_max,
    )
    stack(
        [
            :profit_gain_demand_high_min,
            :profit_gain_demand_high_max,
            :profit_gain_demand_low_min,
            :profit_gain_demand_low_max,
            :profit_gain_demand_all_min,
            :profit_gain_demand_all_max,
        ],
        variable_name = :profit_gain_type,
        value_name = :profit_gain,
    )
    @transform(
        :demand_level =
            replace(:profit_gain_type, r"profit_gain_demand_([a-z]+)_.*" => s"\1")
    )
    @transform(
        :statistic =
            replace(:profit_gain_type, r"profit_gain_demand_[a-z]+_([a-z_]+)" => s"\1")
    )
    @select(
        :statistic,
        :profit_gain,
        :demand_level,
        :weak_signal_quality_level,
        :frequency_high_demand
    )
    @subset(!ismissing(:profit_gain))
    @sort(:frequency_high_demand)
    data(_) *
    mapping(
        :frequency_high_demand => "High Demand Frequency",
        :profit_gain => "Profit Gain",
        row = :statistic => nonnumeric => "Metric",
        color = :demand_level => nonnumeric => "Demand Level",
        col = :weak_signal_quality_level => nonnumeric,
    ) *
    (visual(Lines))
end
# NOTE: freq_high_demand == 1 intersect weak_signal_quality_level == 1 is excluded, as the low demand states are never explored, so the price response to demand signal is not defined
f23 = draw(
    plt23,
    legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
    axis = (
        xticks = 0.0:0.5:1,
        xlabelsize = 0.1,
        yticks = 0:0.2:1.2,
        limits = (0.0, 1.0, 0.0, 1.0),
    ),
)
save("plots/dddc/plot_23.svg", f23)

plt231 = @chain df_summary begin
    @subset(:strong_signal_quality_level == :weak_signal_quality_level)
    @transform(
        :profit_gain_demand_all_min = :profit_gain_min,
        :profit_gain_demand_all_max = :profit_gain_max,
    )
    stack(
        [
            :profit_gain_demand_high_min,
            :profit_gain_demand_high_max,
            :profit_gain_demand_low_min,
            :profit_gain_demand_low_max,
            :profit_gain_demand_all_min,
            :profit_gain_demand_all_max,
        ],
        variable_name = :profit_gain_type,
        value_name = :profit_gain,
    )
    @transform(
        :demand_level =
            replace(:profit_gain_type, r"profit_gain_demand_([a-z]+)_.*" => s"\1")
    )
    @transform(
        :statistic =
            replace(:profit_gain_type, r"profit_gain_demand_[a-z]+_([a-z_]+)" => s"\1")
    )
    data(_) *
    mapping(
        :frequency_high_demand => "High Demand Frequency",
        :weak_signal_quality_level => "Symmetric Signal Strength",
        :profit_gain => "Profit Gain",
        row = :demand_level => nonnumeric => "Demand Level",
        col = :statistic => "Metric",
    ) *
    visual(Heatmap)
end

f231 = draw(
    plt231,
    legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
    axis = (
        xticks = 0.0:0.1:1,
        yticks = 0.5:0.1:1,
        aspect = 1,
        limits = (0.0, 1.0, 0.5, 1.0),
    ),
)
save("plots/dddc/plot_223.svg", f231)

plt24 = @chain df_summary begin
    stack(
        [
            :profit_gain_demand_high_weak_signal_player,
            :profit_gain_demand_low_weak_signal_player,
            :profit_gain_demand_high_strong_signal_player,
            :profit_gain_demand_low_strong_signal_player,
        ],
        variable_name = :profit_gain_type,
        value_name = :profit_gain,
    )
    @subset(:strong_signal_quality_level == strong_signal_level)
    @sort(:frequency_high_demand)
    @transform(
        :demand_level =
            replace(:profit_gain_type, r"profit_gain_demand_([a-z]+)_.*" => s"\1"),
        :weak_signal_quality_level = round(:weak_signal_quality_level; digits = 1),
        :signal_type = replace(
            :profit_gain_type,
            r"profit_gain_demand_[a-z]+_([a-z_]+)_signal_player" => s"\1",
        )
    )
    # @subset(!ismissing(:profit_gain)) # TODO: Remove this once you figure out why missings are in data (or whether they are even in data for fresh runs...)
    @groupby(
        :demand_level,
        :weak_signal_quality_level,
        :strong_signal_quality_level,
        :signal_type,
        :frequency_high_demand
    )
    @combine(:profit_gain = mean(:profit_gain),)
    @subset((:frequency_high_demand < 1) | (:demand_level == "high"))
    data(_) *
    mapping(
        :frequency_high_demand => "High Demand Frequency",
        :profit_gain,
        row = :demand_level => nonnumeric => "Demand Level",
        col = :weak_signal_quality_level => nonnumeric,
        color = :signal_type => "Signal Strength",
    ) *
    (visual(Lines))
end

# NOTE: freq_high_demand == 1 intersect weak_signal_quality_level == 1 is excluded, as the low demand states are never explored, so the price response to demand signal is not defined
f24 = draw(
    plt24,
    legend = (
        position = :top,
        titleposition = :left,
        framevisible = true,
        padding = 5,
        titlesize = 10,
        labelsize = 10,
    ),
    axis = (
        # title = x -> string(x, "aaa"),
        # subtitle = "(Solid line is profit for symmetric prices, shaded region shows range based on price options)",
        xlabel = "High Demand Frequency",
        ylabel = "Profit Gain",
        # xticks = 0.5:0.2:1,
        # yticks = 0:0.2:1,
        # aspect = 0.5,
        # limits = (0.5, 1.01, 0.0, 1.0),
    ),
)
save("plots/dddc/plot_24.svg", f24)

plt25 = @chain df_summary begin
    stack(
        [
            :convergence_profit_demand_high_weak_signal_player,
            :convergence_profit_demand_low_weak_signal_player,
            :convergence_profit_demand_high_strong_signal_player,
            :convergence_profit_demand_low_strong_signal_player,
        ],
        variable_name = :convergence_profit_type,
        value_name = :convergence_profit,
    )
    @subset(
        (:frequency_high_demand != 1) &
        (:weak_signal_quality_level == round(:weak_signal_quality_level; digits = 1)) &
        (:strong_signal_quality_level == strong_signal_level)
    )
    @sort(:frequency_high_demand)
    @transform(
        :convergence_profit_type =
            replace(:convergence_profit_type, "convergence_profit_" => "")
    )
    @transform(:convergence_profit_type = replace(:convergence_profit_type, "_" => " "))
    # @subset(!ismissing(:convergence_profit)) # TODO: Remove this once you figure out why missings are in data (or whether they are even in data for fresh runs...)
    data(_) *
    mapping(
        :frequency_high_demand => "High Demand Frequency",
        :convergence_profit => "Average Profit",
        color = :convergence_profit_type => nonnumeric => "Demand Level",
        layout = :weak_signal_quality_level => nonnumeric,
    ) *
    (visual(Lines))
end
# NOTE: freq_high_demand == 1 intersect weak_signal_quality_level == 1 is excluded, as the low demand states are never explored, so the price response to demand signal is not defined
f25 = draw(
    plt25,
    legend = (
        position = :top,
        titleposition = :left,
        framevisible = true,
        padding = 5,
        titlesize = 10,
        labelsize = 10,
        nbanks = 2,
    ),
    # axis = (width = 100, height = 100),
)
save("plots/dddc/plot_25.svg", f25)

plt26 = @chain df_summary begin
    stack(
        [
            :percent_unexplored_states_weak_signal_player,
            :percent_unexplored_states_strong_signal_player,
        ],
        variable_name = :percent_unexplored_states_type,
        value_name = :percent_unexplored_states_value,
    )
    @subset(
        (:weak_signal_quality_level == round(:weak_signal_quality_level; digits = 1)) &
        (:strong_signal_quality_level == strong_signal_level)
    )
    @sort(:frequency_high_demand)
    @transform(
        :percent_unexplored_states_type = replace(
            :percent_unexplored_states_type,
            "percent_unexplored_states_" => "",
        )
    )
    @transform(
        :percent_unexplored_states_type =
            replace(:percent_unexplored_states_type, "_" => " ")
    )
    @subset(!ismissing(:percent_unexplored_states_value)) # TODO: Remove this once you figure out why missings are in data (or whether they are even in data for fresh runs...)
    data(_) *
    mapping(
        :frequency_high_demand => "High Demand Frequency",
        :percent_unexplored_states_value => "Frequency Unexplored States",
        color = :percent_unexplored_states_type => nonnumeric => "Signal Strength",
        layout = :weak_signal_quality_level => nonnumeric => "Weak Signal Strength",
    ) *
    (visual(Lines))
end
# NOTE: freq_high_demand == 1 intersect weak_signal_quality_level == 1 is excluded, as the low demand states are never explored, so the price response to demand signal is not defined
f26 = draw(
    plt26,
    legend = (
        position = :top,
        titleposition = :left,
        framevisible = true,
        padding = 5,
        titlesize = 10,
        labelsize = 10,
        nbanks = 2,
    ),
    # axis = (yticks = 0:0.000025:0.0001, xticks = 0:0.1:1, limits = (0.5, 1.01, 0.0, 0.0001)),
)
save("plots/dddc/plot_26.svg", f26)

plt27 = @chain df_summary begin
    stack(
        [
            :profit_gain_demand_high_weak_signal_player,
            :profit_gain_demand_low_weak_signal_player,
            :profit_gain_demand_high_strong_signal_player,
            :profit_gain_demand_low_strong_signal_player,
        ],
        variable_name = :profit_gain_type,
        value_name = :profit_gain,
    )
    @subset(
        (:frequency_high_demand != 1) &
        (:weak_signal_quality_level == round(:weak_signal_quality_level; digits = 1)) &
        (:strong_signal_quality_level == strong_signal_level)
    )
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
    @transform(:signal_type = uppercasefirst(:signal_type) * " Signal Player")
    @transform(:demand_level = uppercasefirst(:demand_level) * " Demand")
    @subset(!ismissing(:profit_gain)) # TODO: Remove this once you figure out why missings are in data (or whether they are even in data for fresh runs...)
    data(_) *
    mapping(
        :frequency_high_demand => "High Demand Frequency",
        :profit_gain => "Profit Gain",
        color = :weak_signal_quality_level => nonnumeric => "Weak Signal Strength",
        row = :demand_level => nonnumeric => "Demand Level",
        col = :signal_type => nonnumeric => "Signal Strength",
    ) *
    (visual(Lines))
end
# NOTE: freq_high_demand == 1 intersect weak_signal_quality_level == 1 is excluded, as the low demand states are never explored, so the price response to demand signal is not defined
f27 = draw(
    plt27,
    legend = (
        position = :top,
        titleposition = :left,
        framevisible = true,
        padding = 5,
        titlesize = 10,
        labelsize = 10,
    ),
)
save("plots/dddc/plot_27.svg", f27)

df_weak_weak_outcomes = @chain df_summary begin
    @subset(
        (:strong_signal_quality_level == :weak_signal_quality_level) &
        (:frequency_high_demand < 1.0) &
        (:frequency_high_demand > 0.0) &
        (:weak_signal_quality_level == round(:weak_signal_quality_level; digits = 1))
    )
    @sort(:frequency_high_demand)
end

plt_28 = @chain df_weak_weak_outcomes begin
    data(_) *
    mapping(
        :frequency_high_demand => "High Demand Frequency",
        :pct_compensating_profit_gain => "Frequency of Weak-Weak Outcomes with Compensating Profit Gain",
        color = :weak_signal_quality_level => nonnumeric => "Symmetric Signal Strength",
    ) *
    visual(Lines)

end
f28 = draw(
    plt_28,
    axis = (xticks = 0.0:0.1:1, yticks = 0:0.1:1, limits = (0.0, 1.02, 0.0, 1.0)),
)
save("plots/dddc/plot_28.svg", f28)

plt3 = @chain df_summary begin
    @sort(:frequency_high_demand)
    @transform(
        :signal_is_strong =
            :strong_signal_quality_level == :weak_signal_quality_level ? "Weak-Weak" :
            "Strong-Weak"
    )
    @subset(
        (:weak_signal_quality_level == round(:weak_signal_quality_level; digits = 1)) &
        (:strong_signal_quality_level == strong_signal_level)
    )
    data(_) *
    mapping(
        :frequency_high_demand => "High Demand Frequency",
        :iterations_until_convergence => "Iterations Until Convergence",
        color = :weak_signal_quality_level => nonnumeric => "Symmetric Signal Strength",
        layout = :signal_is_strong => nonnumeric,
    ) *
    visual(Lines)
end
f3 = draw(plt3, axis = (xticks = 0.0:0.1:1,))
save("plots/dddc/plot_3.svg", f3)

freq_high_demand = 0.9
for freq_high_demand = 0.0:0.1:1
    n_bins_ = 200
    df_summary_rounded = @chain df_summary begin
        # @subset(!ismissing(:profit_gain_min)) # TODO: Remove this once you figure out why missings are in data (or whether they are even in data for fresh runs...)
        # @subset(!ismissing(:profit_gain_max)) # TODO: Remove this once you figure out why missings are in data (or
        # @subset(!ismissing(:profit_gain_demand_high_weak_signal_player)) # TODO: Remove this once you figure out why missings are in data (or
        # @subset(!ismissing(:profit_gain_demand_high_strong_signal_player)) # TODO: Remove this once you figure out why missings are in data (or
        # @transform(
        #     :weak_signal_quality_level = round(:weak_signal_quality_level * n_bins_; digits=0) / n_bins_,
        #     :strong_signal_quality_level = round(:strong_signal_quality_level * n_bins_; digits=0) / n_bins_,
        #     :frequency_high_demand = round(:frequency_high_demand * 10; digits=0) / 10,
        # # ) # TODO: Remove this once you figure out why missings are in data (or whether they are even in data for fresh runs...)
        # @groupby(:weak_signal_quality_level, :strong_signal_quality_level, :frequency_high_demand)
        # @combine(
        #     :profit_gain_min = mean(:profit_gain_min),
        #     :profit_gain_max = mean(:profit_gain_max),
        #     :profit_gain_weak_signal_player = mean(:profit_gain_demand_high_weak_signal_player),
        #     :profit_gain_strong_signal_player = mean(:profit_gain_demand_high_strong_signal_player),
        # )
    end
    df_summary_weak_weak = @chain df_summary_rounded begin
        @subset(:frequency_high_demand == freq_high_demand)
        @subset(:weak_signal_quality_level == :strong_signal_quality_level)
        @select(
            :signal_quality_level = :weak_signal_quality_level,
            :profit_gain_avg = (:profit_gain_max + :profit_gain_min) / 2
        ) # (no semantic effect, but double the sample size)
    end

    df_rework = @chain df_summary_rounded begin
        @subset(:strong_signal_quality_level != 1)
        @subset(
            (:frequency_high_demand == freq_high_demand) &
            (:strong_signal_quality_level != :weak_signal_quality_level)
        )
        leftjoin(
            df_summary_weak_weak,
            on = :strong_signal_quality_level => :signal_quality_level,
            renamecols = "" => "_signal_ceil",
        )
        leftjoin(
            df_summary_weak_weak,
            on = :weak_signal_quality_level => :signal_quality_level,
            renamecols = "" => "_signal_floor",
        )
        @transform(
            :profit_gain_delta_strong_player_signal_ceil =
                :profit_gain_strong_signal_player - :profit_gain_avg_signal_ceil,
            :profit_gain_delta_strong_player_signal_floor =
                :profit_gain_strong_signal_player - :profit_gain_avg_signal_floor,
        )
        # @subset(!ismissing(:profit_gain_strong_signal_player)) # TODO: Remove this once you figure out why missings are in data (or whether they are even in data for fresh runs...)
        @transform(
            :profit_gain_delta_weak_player_signal_ceil =
                :profit_gain_weak_signal_player - :profit_gain_avg_signal_ceil,
            :profit_gain_delta_weak_player_signal_floor =
                :profit_gain_weak_signal_player - :profit_gain_avg_signal_floor,
        )
        @transform(
            :weak_player_best_information = argmax([
                :profit_gain_weak_signal_player,
                :profit_gain_avg_signal_ceil,
                :profit_gain_avg_signal_floor,
            ]),
            :strong_player_best_information = argmax([
                :profit_gain_strong_signal_player,
                :profit_gain_avg_signal_ceil,
                :profit_gain_avg_signal_floor,
            ])
        )
        @transform(
            :weak_player_best_information_string =
                :weak_player_best_information == 1 ? "h" :
                :weak_player_best_information == 2 ? "s" : "d",
            :strong_player_best_information_string =
                :strong_player_best_information == 1 ? "h" :
                :strong_player_best_information == 2 ? "s" : "d",
        )
        @transform(
            :weak_player_worst_information = argmin([
                :profit_gain_weak_signal_player,
                :profit_gain_avg_signal_ceil,
                :profit_gain_avg_signal_floor,
            ]),
            :strong_player_worst_information = argmin([
                :profit_gain_strong_signal_player,
                :profit_gain_avg_signal_ceil,
                :profit_gain_avg_signal_floor,
            ])
        )
        @transform(
            :weak_player_worst_information_string =
                :weak_player_worst_information == 1 ? "h" :
                :weak_player_worst_information == 2 ? "s" : "d",
            :strong_player_worst_information_string =
                :strong_player_worst_information == 1 ? "h" :
                :strong_player_worst_information == 2 ? "s" : "d",
        )
        @transform(
            :joint_worst_information =
                :strong_player_worst_information_string *
                :weak_player_worst_information_string
        )
        @transform(
            :joint_best_information =
                :strong_player_best_information_string *
                :weak_player_best_information_string
        )
    end

    plt8_partial = @chain df_rework begin
        stack(
            [
                :profit_gain_delta_strong_player_signal_ceil,
                :profit_gain_delta_strong_player_signal_floor,
                :profit_gain_delta_weak_player_signal_ceil,
                :profit_gain_delta_weak_player_signal_floor,
            ],
            variable_name = :signal_intervention,
            value_name = :profit_gain_delta,
        )
        @transform(:player = occursin("weak", :signal_intervention) ? "weak" : "strong")
        @transform(
            :signal_intervention =
                replace(:signal_intervention, r"profit_gain_delta_.*_player_" => "")
        )
        data(_) * mapping(
            :weak_signal_quality_level,
            :strong_signal_quality_level,
            :profit_gain_delta,
            col = :signal_intervention,
            row = :player,
        )
    end
    plt8 = plt8_partial * visual(Heatmap)
    f8 = draw(plt8) #, axis = (xticks = 0.5:0.1:1,))
    save("plots/dddc/plot_8__freq_high_demand_$freq_high_demand.svg", f8)

    plt8_1 = plt8_partial * contour(levels = 8, labels = true)
    f81 = draw(plt8_1) #, axis = (xticks = 0.5:0.1:1,))
    save("plots/dddc/plot_81__freq_high_demand_$freq_high_demand.svg", f81)

    plt82 = @chain df_rework begin
        data(_) *
        mapping(
            :weak_signal_quality_level,
            :strong_signal_quality_level,
            :joint_best_information,
        ) *
        visual(Heatmap)
    end
    scale_custom_color = scales(
        Color = (;
            categories = [
                "ss" => "Share/Share",
                "dd" => "Discard/Discard",
                "hh" => "Hoard/Hoard",
                "hs" => "Hoard/Share",
                "sh" => "Share/Hoard",
                "dh" => "Discard/Hoard",
                "hd" => "Hoard/Discard",
                "ds" => "Discard/Share",
            ],
        ),
    )
    f82 = draw(plt82, scale_custom_color)
    save("plots/dddc/plot_82__freq_high_demand_$freq_high_demand.svg", f82)

    plt83 = @chain df_rework begin
        data(_) *
        mapping(
            :weak_signal_quality_level,
            :strong_signal_quality_level,
            :joint_worst_information,
        ) *
        visual(Heatmap)
    end
    f83 = draw(plt83, scale_custom_color)
    save("plots/dddc/plot_83__freq_high_demand_$freq_high_demand.svg", f82)
end

df_information_summary = @chain df_summary begin
    stack(
        [:profit_gain_strong_signal_player, :profit_gain_weak_signal_player],
        variable_name = :signal_player,
        value_name = :profit_gain_delta,
    )
    @transform(:signal_player = replace(:signal_player, r"profit_gain_" => ""))
    @groupby(:frequency_high_demand, :signal_player)
    @combine(
        :profit_gain_delta_max = maximum(:profit_gain_delta),
        :profit_gain_delta_min = minimum(:profit_gain_delta),
    )
    @transform(:information_value = :profit_gain_delta_max - :profit_gain_delta_min)
    @sort(:frequency_high_demand)
    @transform(:signal_player = replace(:signal_player, "_" => " "))
    @transform(:signal_player = titlecase(:signal_player))
end

plt9 = @chain df_information_summary begin
    data(_) *
    mapping(
        :frequency_high_demand => "High Demand Frequency",
        :profit_gain_delta_min => "Profit Gain",
        :profit_gain_delta_max => "Profit Gain",
        color = :signal_player => nonnumeric => "Signal Player",
        col = :signal_player => nonnumeric,
    ) *
    visual(Band)
end

f912 = Figure(; size = (800, 600))
subfig = f912[1, 1]
grid = draw!(subfig, plt9, axis = (xticks = 0.0:0.2:1, title = ""))
legend!(f912[1, 2], grid)
titlelayout = GridLayout(f912[0, 1], halign = :left, tellwidth = false)
Label(
    titlelayout[1, 1],
    "Profit Possibilities Range",
    halign = :left,
    fontsize = 20,
    font = "TeX Gyre Heros Bold Makie",
)
rowgap!(titlelayout, 0)
f912
save("plots/dddc/plot_9.svg", f912)

# TODO: Look into using Bayes' Factor instead of frequency high demand for plot...

plt91 = @chain df_information_summary begin
    data(_) *
    mapping(
        :frequency_high_demand => "High Demand Frequency",
        :information_value => "Information Value\n(Max - Min Profit Gain)",
        color = :signal_player => nonnumeric => "Signal Player",
        # col = :signal_player => nonnumeric,
    ) *
    visual(Lines)
end
f91 = draw(
    plt91,
    axis = (xticks = 0.0:0.2:1, title = "Value of Information Set by Demand Setting"),
)
save("plots/dddc/plot_9_1.svg", f91)

function arg_nth_percentile(x, p::Float64)
    partialsortperm(x, Int64(round(length(x) * p, digits = 0)))
end
@test arg_nth_percentile([1.5, 2.5, 3.5], 0.33) == 1


df_profit_max_min_signal_strength = @chain df_summary begin
    @subset((:strong_signal_quality_level <= 0.9) & (:weak_signal_quality_level <= 0.9))
    @groupby(:frequency_high_demand)
    @combine(
        :weak_player__signal_quality_for_profit_max__strong_player_preferences =
            :weak_signal_quality_level[argmax(:profit_gain_strong_signal_player)],
        :weak_player__signal_quality_for_profit_max__weak_player_preferences =
            :weak_signal_quality_level[argmax(:profit_gain_weak_signal_player)],
        :strong_player__signal_quality_for_profit_max__strong_player_preferences =
            :strong_signal_quality_level[argmax(:profit_gain_strong_signal_player)],
        :strong_player__signal_quality_for_profit_max__weak_player_preferences =
            :strong_signal_quality_level[argmax(:profit_gain_weak_signal_player)],
        :weak_player__signal_quality_for_profit_min =
            :weak_signal_quality_level[argmin(
                (:profit_gain_strong_signal_player + :profit_gain_weak_signal_player) / 2,
            )],
        :strong_player__signal_quality_for_profit_min =
            :strong_signal_quality_level[argmin(
                (:profit_gain_strong_signal_player + :profit_gain_weak_signal_player) / 2,
            )],
    )
end

plt_10_1 = @chain df_profit_max_min_signal_strength begin
    stack(
        [
            :weak_player__signal_quality_for_profit_max__strong_player_preferences,
            :weak_player__signal_quality_for_profit_max__weak_player_preferences,
            :strong_player__signal_quality_for_profit_max__strong_player_preferences,
            :strong_player__signal_quality_for_profit_max__weak_player_preferences,
        ],
        variable_name = :player_situation,
        value_name = :signal_quality,
    )
    @transform(:player_situation)
    @transform(:profit_max_for_player = replace(:player_situation, r"^.*__" => ""))
    @transform(:signal_quality_player = replace(:player_situation, r"__.*$" => ""))
    @transform(:signal_quality_player = replace(:signal_quality_player, "_" => " "))
    @transform(:profit_max_for_player = replace(:profit_max_for_player, "_" => " "))
    @transform(
        :profit_max_for_player = replace(:profit_max_for_player, " preferences" => "")
    )
    @sort(:frequency_high_demand)
    data(_) *
    mapping(
        :frequency_high_demand => "High Demand Frequency",
        :signal_quality => "Signal Quality",
        row = :signal_quality_player => titlecase => "",
        color = :profit_max_for_player => titlecase => "Profit Maximizing for:",
    ) *
    visual(Lines)
end


f = Figure(; size = (800, 600))
subfig = f[1, 1]
grid = draw!(subfig, plt_10_1, axis = (xticks = 0.0:0.2:1, yticks = 0.0:0.1:1, title = ""))
legend!(f[1, 2], grid)
titlelayout = GridLayout(f[0, 1], halign = :left, tellwidth = false)
Label(
    titlelayout[1, 1],
    "Profit Maximizing Signal Strengths",
    halign = :left,
    fontsize = 20,
    font = "TeX Gyre Heros Bold Makie",
)
rowgap!(titlelayout, 0)
f
save("plots/dddc/plot_10_1.svg", f)

plt_10_2 = @chain df_profit_max_min_signal_strength begin
    stack(
        [
            :weak_player__signal_quality_for_profit_min,
            :strong_player__signal_quality_for_profit_min,
        ],
        variable_name = :player_situation,
        value_name = :signal_quality,
    )
    @transform(:player_situation)
    @transform(:profit_min_for_product = replace(:player_situation, r"^.*__" => ""))
    @transform(:signal_quality_player = replace(:player_situation, r"__.*$" => ""))
    @transform(:signal_quality_player = replace(:signal_quality_player, "_" => " "))
    @sort(:frequency_high_demand)
    data(_) *
    mapping(
        :frequency_high_demand => "High Demand Frequency",
        :signal_quality => "Signal Quality",
        color = :signal_quality_player => titlecase => "",
    ) *
    visual(Lines)
end
f10_2 = draw(
    plt_10_2,
    axis = (
        xticks = 0.0:0.2:1,
        title = "Profit Minimizing Signal Strength",
        subtitle = "(Signal strength capped at 0.9)",
    ),
)
save("plots/dddc/plot_10_2.svg", f10_2)

# plt4 = @chain df_summary begin
#     @subset(:weak_signal_quality_level == round(:weak_signal_quality_level; digits=1) & (:strong_signal_quality_level == strong_signal_level))
#     data(_) *
#     mapping(
#         :frequency_high_demand => "High Demand Frequency",
#         :profit_min => "Minimum Player Profit per Trial",
#         color = :signal_is_strong => nonnumeric,
#     ) *
#     (visual(Lines))
# end
# f4 = draw(plt4)
# save("plots/dddc/plot_4.svg", f4)

# plt5 = @chain df_summary begin
#     data(_) *
#     mapping(
#         :frequency_high_demand => "High Demand Frequency",
#         :profit_max_mean => "Maximum Player Profit per Trial",
#         color = :signal_is_strong => nonnumeric,
#     ) *
#     (visual(Scatter) + linear())
# end
# f5 = draw(plt5)
# save("plots/dddc/plot_5.svg", f5)

# α = Float64(0.125)
# β = Float64(4e-1)
# δ = 0.95
# ξ = 0.1
# δ = 0.95
# n_prices = 15
# max_iter = Int(1e6) # 1e8
# price_index = 1:n_prices

# competition_params_dict = Dict(
#     :high => CompetitionParameters(0.25, -0.25, (2, 2), (1, 1)),
#     :low => CompetitionParameters(0.25, 0.25, (2, 2), (1, 1)),
# )

# competition_solution_dict =
#     Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

# data_demand_digital_params = DataDemandDigitalParams(
#     weak_signal_quality_level = 0.99,
#     strong_signal_quality_level = 0.995,
#     signal_is_strong = [true, false],
#     frequency_high_demand = 0.9,
# )

# hyperparams = DDDCHyperParameters(
#     α,
#     β,
#     δ,
#     max_iter,
#     competition_solution_dict,
#     data_demand_digital_params;
#     convergence_threshold = Int(1e5),
# )

# plt = draw_price_diagnostic(hyperparams)
# f6 = draw(
#     plt,
#     axis = (
#         title = "Profit Levels across Price Options",
#         subtitle = "(Solid line is profit for symmetric prices, shaded region shows range based on price options)",
#         xlabel = "Competitor's Price Choice",
#     ),
#     legend = (position = :bottom, titleposition = :left, framevisible = true, padding = 5),
# )
# save("plots/dddc/plot_6.svg", f6)
