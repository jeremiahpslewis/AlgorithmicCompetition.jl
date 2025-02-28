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
df_summary_arrow_cache_path = "data_final/dddc_v0.1.1_data_summary.arrow"

arrow_folders = filter!(
    x -> occursin(r"model=dddc_version=2025-01-20-dddc-revised-prices", x),
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
    df_summary = AlgorithmicCompetition.reduce_dddc(df_summary)
    mkpath("data_final")
    Arrow.write(df_summary_arrow_cache_path, df_summary)
end

mkpath("plots/dddc")

strong_signal_level = 1.0
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
        (:strong_signal_quality_level == :weak_signal_quality_level)
    #     (:weak_signal_quality_level == round(:weak_signal_quality_level; digits = 1))
    )
    @sort(:weak_signal_quality_level)
    data(_) *
    mapping(
        :weak_signal_quality_level => "Symmetric Signal Strength",
        :profit_mean => "Average Profit",
        color = :frequency_high_demand => nonnumeric => "High Demand Frequency",
        # layout = :frequency_high_demand => nonnumeric,
    ) *
    visual(Lines)
end
f20 = draw(
    plt20,
    legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
)
save("plots/dddc/plot_20.svg", f20)

# plt201 = @chain df_summary begin
#     @subset(:strong_signal_quality_level == :weak_signal_quality_level)
#     data(_) *
#     mapping(
#         :frequency_high_demand => "High Demand Frequency",
#         :weak_signal_quality_level => "Symmetric Signal Strength",
#         :profit_mean => "Average Profit",
#     ) *
#     visual(Heatmap)
# end
# f201 = draw(
#     plt201,
#     legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
# )
# save("plots/dddc/plot_201.svg", f201)


plt21 = @chain df_summary begin
    @subset(
        (:strong_signal_quality_level == :weak_signal_quality_level) &
        !ismissing(:price_response_to_demand_signal_mse)    )
    @sort(:weak_signal_quality_level)
    data(_) *
    mapping(
        :weak_signal_quality_level => "Symmetric Signal Strength",
        :price_response_to_demand_signal_mse => "Mean Squared Error Price Difference by Demand Signal",
        color = :frequency_high_demand => nonnumeric => "High Demand Frequency",
    ) *
    visual(Lines)
end
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
    @sort(:weak_signal_quality_level)
    data(_) *
    mapping(
        :weak_signal_quality_level           => "Symmetric Signal Strength",
        :frequency_high_demand               => "High Demand Frequency",
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
        (:strong_signal_quality_level == :weak_signal_quality_level) & !ismissing(:profit))
    @transform(:demand_level = replace(:demand_level, "convergence_profit_demand_" => ""))
    @sort(:weak_signal_quality_level)
    @transform(
        :demand_level_str = string("Demand Level: ", :demand_level)
    )
    data(_) *
    mapping(
        :weak_signal_quality_level => "Signal Strength",
        :profit                      => "Average Profit",
        color = :demand_level_str        => nonnumeric => "Demand Level",
        layout = :frequency_high_demand => nonnumeric => "High Demand Frequency",
    ) *
    visual(Lines)
end
f22 = draw(
    plt22,
    legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
)
save("plots/dddc/plot_22.svg", f22)

plt22aa = @chain df_summary begin
    @transform(
        :profit_gain_demand_high = (:profit_gain_demand_high_max + :profit_gain_demand_high_min) / 2,
        :profit_gain_demand_low = (:profit_gain_demand_low_max + :profit_gain_demand_low_min) / 2,
    )
    stack(
        [:profit_gain_demand_high, :profit_gain_demand_low],
        variable_name = :demand_level,
        value_name = :profit_gain,
    )
    @subset(
        (:strong_signal_quality_level == :weak_signal_quality_level) & !ismissing(:profit_gain))
    @transform(:demand_level = replace(:demand_level, "profit_gain_demand_" => ""))
    @sort(:weak_signal_quality_level)
    @transform(
        :demand_level_str = string("Demand Level: ", :demand_level)
    )
    data(_) *
    mapping(
        :weak_signal_quality_level => "Signal Strength",
        :profit_gain               => "Profit Gain",
        color = :demand_level_str  => nonnumeric => "Demand Level",
        row = :frequency_high_demand => nonnumeric => "High Demand Frequency",
    ) *
    visual(Lines)
end
f22aa = draw(
    plt22aa,
    legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
)
save("plots/dddc/plot_22aa.svg", f22aa)

# Mean, max, mean, symmetric
plt22aaa = @chain df_summary begin
    @transform(
        :profit_gain_demand_high = (:profit_gain_demand_high_max + :profit_gain_demand_high_min) / 2,
        :profit_gain_demand_low = (:profit_gain_demand_low_max + :profit_gain_demand_low_min) / 2,
    )
    stack(
        [:profit_gain_demand_high, :profit_gain_demand_low],
        variable_name = :demand_level,
        value_name = :profit_gain,
    )
    @subset(!ismissing(:profit_gain))
    @transform(:min_signal_quality_level = min(:strong_signal_quality_level, :weak_signal_quality_level), :max_signal_quality_level = max(:strong_signal_quality_level, :weak_signal_quality_level), :mean_signal_quality_level = (:strong_signal_quality_level + :weak_signal_quality_level) / 2)
    stack(
        [:min_signal_quality_level, :max_signal_quality_level, :mean_signal_quality_level],
        variable_name = :signal_quality_level_type,
        value_name = :signal_quality_level,
    )
    @groupby(:demand_level, :frequency_high_demand, :signal_quality_level_type, :signal_quality_level)
    @combine(:profit_gain = mean(:profit_gain))
    @transform(:demand_level = replace(:demand_level, "profit_gain_demand_" => ""))
    @sort(:signal_quality_level)
    @transform(
        :demand_level_str = string("Demand Level: ", :demand_level)
    )
    data(_) *
    mapping(
        :signal_quality_level => "Signal Strength",
        :profit_gain               => "Profit Gain",
        color = :signal_quality_level_type => "Signal Strength",
        col = :demand_level_str  => nonnumeric => "Demand Level",
        row = :frequency_high_demand => nonnumeric => "High Demand Frequency",
    ) *
    visual(Lines)
end
f22aaa = draw(
    plt22aaa,
    legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
)
save("plots/dddc/plot_22aaa.svg", f22aa)

plt22a = @chain df_summary begin
    @subset((:frequency_high_demand == 0.5))
    @transform(
        :profit_gain_demand_high = (:profit_gain_demand_high_max + :profit_gain_demand_high_min) / 2,
        :profit_gain_demand_low = (:profit_gain_demand_low_max + :profit_gain_demand_low_min) / 2,
    )
    stack(
        [:profit_gain_demand_high, :profit_gain_demand_low],
        variable_name = :demand_level,
        value_name = :profit_gain,
    )
    @transform(:demand_level = replace(:demand_level, "profit_gain_demand_" => ""))
    @sort(:weak_signal_quality_level)
    @transform(
        :demand_level_str = string("Demand Level: ", :demand_level)
    )
    @subset(:strong_signal_quality_level == round(:strong_signal_quality_level, digits = 1))
    data(_) *
    mapping(
        :weak_signal_quality_level => "Weak Signal Strength",
        :profit_gain                      => "Average Profit",
        color = :strong_signal_quality_level        => nonnumeric => "Strong Signal Strength",
        row = :demand_level_str => nonnumeric => "High Demand Frequency",
    ) *
    visual(Lines)
end
f22a = draw(
    plt22a,
    legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
)
save("plots/dddc/plot_22a.svg", f22a)

# plt222 = @chain df_summary begin
#     stack(
#         [:convergence_profit_demand_high, :convergence_profit_demand_low],
#         variable_name = :demand_level,
#         value_name = :profit,
#     )
#     @subset(:strong_signal_quality_level == :weak_signal_quality_level)
#     @transform(
#         :demand_level =
#             replace(:demand_level, "convergence_profit_demand_" => "Demand: ")
#     )
#     data(_) *
#     mapping(
#         :frequency_high_demand => "High Demand Frequency",
#         :weak_signal_quality_level => "Symmetric Signal Strength",
#         :profit => "Average Profit",
#         layout = :demand_level => nonnumeric => "Demand Level",
#     ) *
#     visual(Heatmap)
# end
# f222 = draw(
#     plt222,
#     # legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
# )
# save("plots/dddc/plot_222.svg", f222)

plt221 = @chain df_summary begin
    @subset(
        (:strong_signal_quality_level == :weak_signal_quality_level)
    )
    stack(
        [:profit_gain_min, :profit_gain_max],
        variable_name = :min_max,
        value_name = :profit_gain,
    )
    @transform(
        :min_max = (:min_max == "profit_gain_min" ? "Per-Trial Min" : "Per-Trial Max")
    )
    @sort(:weak_signal_quality_level)
    data(_) *
    mapping(
        :weak_signal_quality_level => "Symmetric Signal Strength",
        :profit_gain,
        color = :min_max => nonnumeric => "",
        layout = :frequency_high_demand => nonnumeric,
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

# plt223 = @chain df_summary begin
#     @subset(:strong_signal_quality_level == :weak_signal_quality_level)
#     stack(
#         [:profit_gain_min, :profit_gain_max],
#         variable_name = :min_max,
#         value_name = :profit_gain,
#     )
#     @transform(
#         :min_max = (:min_max == "profit_gain_min" ? "Per-Trial Min" : "Per-Trial Max")
#     )
#     data(_) *
#     mapping(
#         :frequency_high_demand => "High Demand Frequency",
#         :weak_signal_quality_level => "Symmetric Signal Strength",
#         :profit_gain => "Profit Gain",
#         layout = :min_max => nonnumeric => "",
#     ) *
#     visual(Heatmap)
# end
# f223 = draw(
#     plt223,
#     axis = (xticks = 0.0:0.2:1,),
#     legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
# )
# save("plots/dddc/plot_223.svg", f223)

plt23 = @chain df_summary begin
    @subset(
        (:strong_signal_quality_level == :weak_signal_quality_level)
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
    @sort(:weak_signal_quality_level)
    data(_) *
    mapping(
        :weak_signal_quality_level => "High Demand Frequency",
        :profit_gain => "Profit Gain",
        color = :statistic => nonnumeric => "Metric",
        row = :demand_level => nonnumeric => "Demand Level",
        col = :frequency_high_demand => nonnumeric,
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
        limits = (-0.1, 1.1, -0.1, 1.1),
    ),
)
save("plots/dddc/plot_23.svg", f23)

# plt231 = @chain df_summary begin
#     @subset(:strong_signal_quality_level == :weak_signal_quality_level)
#     @transform(
#         :profit_gain_demand_all_min = :profit_gain_min,
#         :profit_gain_demand_all_max = :profit_gain_max,
#     )
#     stack(
#         [
#             :profit_gain_demand_high_min,
#             :profit_gain_demand_high_max,
#             :profit_gain_demand_low_min,
#             :profit_gain_demand_low_max,
#             :profit_gain_demand_all_min,
#             :profit_gain_demand_all_max,
#         ],
#         variable_name = :profit_gain_type,
#         value_name = :profit_gain,
#     )
#     @transform(
#         :demand_level =
#             replace(:profit_gain_type, r"profit_gain_demand_([a-z]+)_.*" => s"\1")
#     )
#     @transform(
#         :statistic =
#             replace(:profit_gain_type, r"profit_gain_demand_[a-z]+_([a-z_]+)" => s"\1")
#     )
#     data(_) *
#     mapping(
#         :frequency_high_demand => "High Demand Frequency",
#         :weak_signal_quality_level => "Symmetric Signal Strength",
#         :profit_gain => "Profit Gain",
#         row = :demand_level => nonnumeric => "Demand Level",
#         col = :statistic => "Metric",
#     ) *
#     visual(Heatmap)
# end

# f231 = draw(
#     plt231,
#     legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
#     axis = (
#         xticks = 0.0:0.1:1,
#         yticks = 0.5:0.1:1,
#         aspect = 1,
#         limits = (0.0, 1.0, 0.5, 1.0),
#     ),
# )
# save("plots/dddc/plot_223.svg", f231)

df_asym = @chain df_summary begin
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
    @sort(:weak_signal_quality_level)
    @transform(
        :demand_level =
            replace(:profit_gain_type, r"profit_gain_demand_([a-z]+)_.*" => s"\1"),
        :weak_signal_quality_level = round(:weak_signal_quality_level; digits = 1),
        :signal_type = replace(
            :profit_gain_type,
            r"profit_gain_demand_[a-z]+_([a-z_]+)_signal_player" => s"\1",
        )
    )
    @subset(!ismissing(:profit_gain)) # TODO: Remove this once you figure out why missings are in data (or whether they are even in data for fresh runs...)
    @groupby(
        :demand_level,
        :weak_signal_quality_level,
        :strong_signal_quality_level,
        :signal_type,
        :frequency_high_demand
    )
    @combine(:profit_gain = mean(:profit_gain),)
end

for strong_signal_level in [0.9, 1.0]
    plt24 = @chain df_asym begin
        @subset(:strong_signal_quality_level == strong_signal_level)
        data(_) *
        mapping(
            :weak_signal_quality_level => "Weak Signal Strength",
            :profit_gain,
            row = :demand_level => nonnumeric => "Demand Level",
            col = :frequency_high_demand => nonnumeric,
            color = :signal_type => "Player",
        ) *
        (visual(Lines))
    end

    # NOTE: freq_high_demand == 1 intersect weak_signal_quality_level == 1 is excluded, as the low demand states are never explored, so the price response to demand signal is not defined
    f24 = draw(
        plt24,
        # legend = (
        #     position = :top,
        #     titleposition = :left,
        #     framevisible = true,
        #     padding = 5,
        #     titlesize = 10,
        #     labelsize = 10,
        # ),
        axis = (
            # title = x -> string(x, "aaa"),
            # subtitle = "(Solid line is profit for symmetric prices, shaded region shows range based on price options)",
            # xlabel = "High Demand Frequency",
            ylabel = "Profit Gain",
            xticks = 0.5:0.25:1,
            # yticks = 0:0.2:1,
            # aspect = 0.9,
            limits = (0.5, 1.05, -0.05, 1.05),
        ),
    )
    save("plots/dddc/plot_24$strong_signal_level.svg", f24)
end

plt241 = @chain df_asym begin
    @subset(:frequency_high_demand == 0.5)
    @subset(:strong_signal_quality_level == round(:strong_signal_quality_level, digits = 1))
    data(_) *
    mapping(
        :weak_signal_quality_level => "Weak Signal Strength",
        :profit_gain,
        row = :demand_level => nonnumeric => "Demand Level",
        col = :strong_signal_quality_level => nonnumeric => "Strong Signal Strength",
        color = :signal_type => nonnumeric => "Player",
    ) *
    (visual(Lines))
end

# NOTE: freq_high_demand == 1 intersect weak_signal_quality_level == 1 is excluded, as the low demand states are never explored, so the price response to demand signal is not defined
f241 = draw(
    plt241,
    axis = (
        ylabel = "Profit Gain",
        xticks = 0.5:0.25:1,
    ),
)
save("plots/dddc/plot_241.svg", f241)

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

freq_high_demand = 0.5
for freq_high_demand = [0.0, 0.5, 1.0]
    n_bins_ = 200
    df_summary_rounded = df_summary
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
            :profit_gain_delta_strong_player_signal_level_up =
                :profit_gain_avg_signal_ceil - :profit_gain_strong_signal_player,
            :profit_gain_delta_strong_player_signal_level_down =
            :profit_gain_avg_signal_floor - :profit_gain_strong_signal_player,
        )
        # @subset(!ismissing(:profit_gain_strong_signal_player)) # TODO: Remove this once you figure out why missings are in data (or whether they are even in data for fresh runs...)
        @transform(
            :profit_gain_delta_weak_player_signal_level_up =
                :profit_gain_avg_signal_ceil - :profit_gain_weak_signal_player,
            :profit_gain_delta_weak_player_signal_level_down =
                :profit_gain_avg_signal_floor - :profit_gain_weak_signal_player,
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
                :profit_gain_delta_strong_player_signal_level_up,
                :profit_gain_delta_strong_player_signal_level_down,
                :profit_gain_delta_weak_player_signal_level_up,
                :profit_gain_delta_weak_player_signal_level_down,
            ],
            variable_name = :signal_intervention,
            value_name = :profit_gain_delta,
        )
        @transform(:player = occursin("weak", :signal_intervention) ? "weak" : "strong")
        @transform(
            :signal_intervention =
                replace(:signal_intervention, r"profit_gain_delta_.*_player_signal_" => "")
        )
        @transform(:signal_intervention = replace(:signal_intervention, "_" => " "))
        @transform(:signal_intervention = titlecase(:signal_intervention))
        @transform(:profit_gain_delta_pp = :profit_gain_delta * 100)
        @transform(:player = titlecase(:player) * " Player")
        data(_) * mapping(
            :weak_signal_quality_level => "Weak Signal Strength",
            :strong_signal_quality_level => "Strong Signal Strength",
            :profit_gain_delta_pp => "Change in Profit Gain (p.p.)",
            col = :signal_intervention,
            row = :player => titlecase,
        )
    end
    plt8 = plt8_partial * visual(Heatmap)

    f8 = draw(plt8, figure = (; title = "Effect of Signal 'Leveling' on Competition", subtitle = "(High Demand Freq. $freq_high_demand)"))
    save("plots/dddc/plot_8__freq_high_demand_$freq_high_demand.svg", f8)

    plt8_1 = plt8_partial * visual(Contour; levels = 4, labels = false)
    f81 = draw(
        plt8_1,
        figure = (
            title = "Effect of Signal 'Leveling' on Competition",
            subtitle = "(High Demand Freq. $freq_high_demand)",
        ),
        axis = (xticks = 0.0:0.2:1,)
    )
    save("plots/dddc/plot_81__freq_high_demand_$freq_high_demand.svg", f81)

    plt82 = @chain df_rework begin
        @transform(
            :joint_best_information =
                ((:joint_best_information != "ss") & (:joint_best_information != "dd")) ?
                    "Disagree" : :joint_best_information
        )
        data(_) *
        mapping(
            :weak_signal_quality_level => "Weak Signal Strength",
            :strong_signal_quality_level => "Strong Signal Strength",
            :joint_best_information => "Best Intervention for Firms",
        ) *
        visual(Heatmap)
    end
    scale_custom_color = scales(
        Color = (;
            categories = [
                "ss" => "Agree: Level Up",
                "dd" => "Agree: Level Down",
                # "hh" => "Hoard/Hoard",
                # "hs" => "Hoard/Share",
                # "sh" => "Share/Hoard",
                # "dh" => "Discard/Hoard",
                # "hd" => "Hoard/Discard",
                # "ds" => "Discard/Share",
                "Disagree" => "Disagree",
            ],
        ),
    )
    f82 = draw(plt82, figure = (; title = "Best Intervention for Firms", subtitle="(High Demand Freq. $freq_high_demand)"))
    
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
    f83 = draw(plt83)
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
f912 = draw(
    plt9,
    figure = (; size = (800, 600), title = "Profit Possibilities Range"),
    axis = (xticks = 0.0:0.2:1, title = "")
)
save("plots/dddc/plot_9.svg", f912)

df_information_summary_a = @chain df_summary begin
    # @subset(:frequency_high_demand == 0.5)
    stack(
        [:profit_gain_strong_signal_player, :profit_gain_weak_signal_player],
        variable_name = :signal_player,
        value_name = :profit_gain_delta,
    )
    @transform(:signal_player = replace(:signal_player, r"profit_gain_" => ""))
    @groupby(:frequency_high_demand, :strong_signal_quality_level, :signal_player)
    @combine(
        :profit_gain_delta_max = maximum(:profit_gain_delta),
        :profit_gain_delta_min = minimum(:profit_gain_delta),
    )
    @transform(:information_value = :profit_gain_delta_max - :profit_gain_delta_min)
    @sort(:strong_signal_quality_level)
    @transform(:signal_player = replace(:signal_player, "_" => " "))
    @transform(:signal_player = titlecase(:signal_player))
end

plt9a = @chain df_information_summary_a begin
    data(_) *
    mapping(
        :strong_signal_quality_level => "Strong Signal Strength",
        :profit_gain_delta_min => "Profit Gain",
        :profit_gain_delta_max => "Profit Gain",
        color = :signal_player => nonnumeric => "Signal Player",
        col = :signal_player => nonnumeric,
        row = :frequency_high_demand => nonnumeric => "High Demand Frequency",
    ) *
    visual(Band)
end
f912a = draw(
    plt9a,
    figure = (; size = (800, 600), title = "Profit Possibilities Range"),
    axis = (xticks = 0.0:0.2:1, title = "")
)
save("plots/dddc/plot_9a.svg", f912a)


df_information_summary_b = @chain df_summary begin
    # @subset(:frequency_high_demand == 0.5)
    stack(
        [:profit_gain_demand_high_weak_signal_player, 
        :profit_gain_demand_low_weak_signal_player,
        :profit_gain_demand_high_strong_signal_player,
        :profit_gain_demand_low_strong_signal_player],
        variable_name = :demand_signal_player,
        value_name = :profit_gain_delta,
    )
    @transform(:signal_player = replace(:demand_signal_player, r"profit_gain_demand_[^_]+_" => ""))
    @transform(:demand_level = replace(:demand_signal_player, r"profit_gain_demand_([a-z]+)_.*" => s"\1"))
    @groupby(:frequency_high_demand, :strong_signal_quality_level, :signal_player, :demand_level)
    @combine(
        :profit_gain_delta_max = maximum(:profit_gain_delta),
        :profit_gain_delta_min = minimum(:profit_gain_delta),
    )
    @transform(:information_value = :profit_gain_delta_max - :profit_gain_delta_min)
    @sort(:strong_signal_quality_level)
    @transform(:signal_player = replace(:signal_player, "_" => " "))
    @transform(:signal_player = titlecase(:signal_player))
    @transform(:demand_level_player_type = :demand_level * " " * :signal_player)
    @subset(!ismissing(:information_value))
end

plt9b = @chain df_information_summary_b begin
    data(_) *
    mapping(
        :strong_signal_quality_level => "Strong Signal Strength",
        :profit_gain_delta_min => "Profit Gain",
        :profit_gain_delta_max => "Profit Gain",
        color = :signal_player => nonnumeric => "Signal Player",
        col = :demand_level_player_type => nonnumeric,
        row = :frequency_high_demand => nonnumeric => "High Demand Frequency",
    ) *
    visual(Band)
end
f912b = draw(
    plt9b,
    figure = (; size = (800, 600), title = "Profit Possibilities Range"),
    axis = (xticks = 0.0:0.2:1, title = "")
)
save("plots/dddc/plot_9b.svg", f912b)


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


f = draw(
    plt_10_1,
    figure = (size = (800, 600), title = "Profit Maximizing Signal Strengths"),
    axis = (xticks = 0.0:0.2:1, yticks = 0.0:0.1:1),
    legend = (position = :right, title = "")
)
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
        # subtitle = "(Signal strength capped at 0.9)",
    ),
)
save("plots/dddc/plot_10_2.svg", f10_2)


df_profit_max_min_signal_strong = @chain df_summary begin
    @groupby(:strong_signal_quality_level)
    @combine(
        :weak_player__signal_quality_for_profit_max__strong_player_preferences =
            :weak_signal_quality_level[argmax(:profit_gain_strong_signal_player)],
        :weak_player__signal_quality_for_profit_max__weak_player_preferences =
            :weak_signal_quality_level[argmax(:profit_gain_weak_signal_player)],
        :weak_player__signal_quality_for_profit_min =
            :weak_signal_quality_level[argmin(
                (:profit_gain_strong_signal_player + :profit_gain_weak_signal_player) / 2,
            )],
    )
end

df_profit_max_min_signal_weak = @chain df_summary begin
    @groupby(:weak_signal_quality_level)
    @combine(
        :strong_player__signal_quality_for_profit_max__strong_player_preferences =
            :strong_signal_quality_level[argmax(:profit_gain_strong_signal_player)],
        :strong_player__signal_quality_for_profit_max__weak_player_preferences =
            :strong_signal_quality_level[argmax(:profit_gain_weak_signal_player)],
        :strong_player__signal_quality_for_profit_min =
            :strong_signal_quality_level[argmin(
                (:profit_gain_strong_signal_player + :profit_gain_weak_signal_player) / 2,
            )],
    )
end

plt_10_3 = @chain df_profit_max_min_signal_strong begin
    stack(
        [
            :weak_player__signal_quality_for_profit_max__strong_player_preferences,
            :weak_player__signal_quality_for_profit_max__weak_player_preferences,
            :weak_player__signal_quality_for_profit_min,
        ],
        variable_name = :player_situation,
        value_name = :signal_quality_opponent,
    )
    @transform(:player_situation = replace(:player_situation, "weak_player__signal_quality_for_profit_" => ""))
    @sort(:strong_signal_quality_level)
    data(_) *
    mapping(
        :strong_signal_quality_level => "Signal Quality",
        :signal_quality_opponent => "Opponent's Signal Quality",
        color = :player_situation => titlecase => "",
    ) *
    visual(Lines)
end
f10_3 = draw(
    plt_10_3,
    axis = (
        xticks = 0.0:0.2:1,
        title = "Profit Minimizing Signal Strength",
        # subtitle = "(Signal strength capped at 0.9)",
    ),
)
save("plots/dddc/plot_10_3.svg", f10_3)

plt_10_4 = @chain df_profit_max_min_signal_weak begin
    stack(
        [
            :strong_player__signal_quality_for_profit_max__strong_player_preferences,
            :strong_player__signal_quality_for_profit_max__weak_player_preferences,
            :strong_player__signal_quality_for_profit_min,
        ],
        variable_name = :player_situation,
        value_name = :signal_quality_opponent,
    )
    @transform(:player_situation = replace(:player_situation, "strong_player__signal_quality_for_profit_" => ""))
    @sort(:weak_signal_quality_level)
    data(_) *
    mapping(
        :weak_signal_quality_level => "Signal Quality",
        :signal_quality_opponent => "Opponent's Signal Quality",
        color = :player_situation => titlecase => "",
    ) *
    visual(Lines)
end
f10_4 = draw(
    plt_10_4,
    axis = (
        xticks = 0.0:0.2:1,
        title = "Profit Minimizing Signal Strength (Flipped)",
        # subtitle = "(Signal strength capped at 0.9)",
    ),
)
save("plots/dddc/plot_10_4.svg", f10_4)


df_profit_by_weak_signal_level = @chain df_summary begin
    @groupby(:weak_signal_quality_level)
    @combine(
        :profit_gain_strong_10pct = quantile(:profit_gain_strong_signal_player, 0.1),
        :profit_gain_strong_90pct = quantile(:profit_gain_strong_signal_player, 0.9),
        :profit_gain_weak_10pct   = quantile(:profit_gain_weak_signal_player, 0.1),
        :profit_gain_weak_90pct   = quantile(:profit_gain_weak_signal_player, 0.9),
        :profit_gain_avg_10pct    = quantile((:profit_gain_strong_signal_player + :profit_gain_weak_signal_player) / 2, 0.1),
    )
end


df_summary_weak_signal_summary = @chain df_summary begin
    leftjoin(
        df_profit_by_weak_signal_level,
        on = :weak_signal_quality_level => :weak_signal_quality_level,
    )
    @groupby(:weak_signal_quality_level)
    @combine(
        :signal_for_strong_player_profit_max_lower_bound = minimum(:strong_signal_quality_level[:profit_gain_strong_signal_player .>= :profit_gain_strong_90pct]),
        :signal_for_strong_player_profit_max_upper_bound = maximum(:strong_signal_quality_level[:profit_gain_strong_signal_player .>= :profit_gain_strong_90pct]),
        :signal_for_weak_player_profit_max_lower_bound   = minimum(:strong_signal_quality_level[:profit_gain_weak_signal_player .>= :profit_gain_weak_90pct]),
        :signal_for_weak_player_profit_max_upper_bound   = maximum(:strong_signal_quality_level[:profit_gain_weak_signal_player .>= :profit_gain_weak_90pct]),
        :signal_for_profit_min_upper_bound  = maximum(:strong_signal_quality_level[(:profit_gain_weak_signal_player + :profit_gain_strong_signal_player) / 2 .<= :profit_gain_avg_10pct]),
        :signal_for_profit_min_lower_bound  = minimum(:strong_signal_quality_level[(:profit_gain_weak_signal_player + :profit_gain_strong_signal_player) / 2 .<= :profit_gain_avg_10pct]),
    )
end


# Create a row for the weak player's bounds (based on weak signal quality grouping)
df_lower = @chain df_summary_weak_signal_summary begin
    @select(
        :weak_signal_quality_level,
        :lower_bound = :signal_for_weak_player_profit_max_lower_bound,
        :upper_bound = :signal_for_weak_player_profit_max_upper_bound
    )
    @transform(:player = "Weak Player")
end

# Create a row for the strong player's bounds (based on weak signal quality grouping)
df_strong = @chain df_summary_weak_signal_summary begin
    @select(
        :weak_signal_quality_level,
        :lower_bound = :signal_for_strong_player_profit_max_lower_bound,
        :upper_bound = :signal_for_strong_player_profit_max_upper_bound
    )
    @transform(:player = "Strong Player")
end

# Combine the two sets of rows (for weak signal quality grouping)
df_combined = vcat(df_lower, df_strong)

plt_11_1 = @chain df_combined begin
    @sort(:weak_signal_quality_level)
    data(_) *
    mapping(
        :weak_signal_quality_level  => "Weak Signal Strength",
        :lower_bound,
        lower = :lower_bound         => "Lower Bound",
        upper = :upper_bound         => "Upper Bound",
        color = :player              => nonnumeric => "Profit Maximizing for:"
    ) *
    visual(LinesFill)
end

f11_1 = draw(
    plt_11_1,
    axis = (
        xticks = 0.5:0.1:1,
        yticks = 0.5:0.1:1,
        title = "Signal Strength for Profit Maximization"
    )
)

################################################################################
# Now add the same thing for strong_signal_quality_level

df_profit_by_strong_signal_level = @chain df_summary begin
    @groupby(:strong_signal_quality_level)
    @combine(
        :profit_gain_strong_10pct = quantile(:profit_gain_strong_signal_player, 0.1),
        :profit_gain_strong_90pct = quantile(:profit_gain_strong_signal_player, 0.9),
        :profit_gain_weak_10pct   = quantile(:profit_gain_weak_signal_player, 0.1),
        :profit_gain_weak_90pct   = quantile(:profit_gain_weak_signal_player, 0.9),
        :profit_gain_avg_10pct    = quantile((:profit_gain_strong_signal_player + :profit_gain_weak_signal_player) / 2, 0.1),        
    )
end


df_summary_strong_signal_summary = @chain df_summary begin
    leftjoin(
        df_profit_by_strong_signal_level,
        on = :strong_signal_quality_level => :strong_signal_quality_level,
    )
    @groupby(:strong_signal_quality_level)
    @combine(
        :signal_for_strong_player_profit_max_lower_bound = minimum(:weak_signal_quality_level[:profit_gain_strong_signal_player .>= :profit_gain_strong_90pct]),
        :signal_for_strong_player_profit_max_upper_bound = maximum(:weak_signal_quality_level[:profit_gain_strong_signal_player .>= :profit_gain_strong_90pct]),
        :signal_for_weak_player_profit_max_lower_bound   = minimum(:weak_signal_quality_level[:profit_gain_weak_signal_player .>= :profit_gain_weak_90pct]),
        :signal_for_weak_player_profit_max_upper_bound   = maximum(:weak_signal_quality_level[:profit_gain_weak_signal_player .>= :profit_gain_weak_90pct]),
        :signal_for_profit_min_upper_bound  = maximum(:weak_signal_quality_level[(:profit_gain_weak_signal_player + :profit_gain_strong_signal_player) / 2 .<= :profit_gain_avg_10pct]),
        :signal_for_profit_min_lower_bound  = minimum(:weak_signal_quality_level[(:profit_gain_weak_signal_player + :profit_gain_strong_signal_player) / 2 .<= :profit_gain_avg_10pct]),
    )
end

# Create a row for the weak player's bounds (based on strong signal quality grouping)
df_weak = @chain df_summary_strong_signal_summary begin
    @select(
        :strong_signal_quality_level,
        :lower_bound = :signal_for_weak_player_profit_max_lower_bound,
        :upper_bound = :signal_for_weak_player_profit_max_upper_bound
    )
    @transform(:player = "Weak Player")
end

# Create a row for the strong player's bounds (based on strong signal quality grouping)
df_strong = @chain df_summary_strong_signal_summary begin
    @select(
        :strong_signal_quality_level,
        :lower_bound = :signal_for_strong_player_profit_max_lower_bound,
        :upper_bound = :signal_for_strong_player_profit_max_upper_bound
    )
    @transform(:player = "Strong Player")
end

df_min = @chain df_summary_strong_signal_summary begin
    @select(
        :strong_signal_quality_level,
        :lower_bound = :signal_for_profit_min_lower_bound,
        :upper_bound = :signal_for_profit_min_upper_bound
    )
    @transform(:player = "Profit Minimizer")
end

# Combine the two sets of rows (for strong signal quality grouping)
df_combined_strong = vcat(df_weak, df_strong, df_min)

plt_11_2 = @chain df_combined_strong begin
    @sort(:strong_signal_quality_level)
    data(_) *
    mapping(
        :strong_signal_quality_level  => "Strong Signal Strength",
        :lower_bound,
        lower = :lower_bound           => "Lower Bound",
        upper = :upper_bound           => "Upper Bound",
        color = :player                => nonnumeric => "Profit Maximizing for:"
    ) *
    visual(LinesFill)
end

f11_2 = draw(
    plt_11_2,
    axis = (
        xticks = 0.5:0.1:1,
        yticks = 0.5:0.1:1,
        title = "Signal Strength for Profit Maximization (Strong Signals)"
    )
)
