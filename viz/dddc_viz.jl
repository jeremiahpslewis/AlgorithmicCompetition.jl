using CairoMakie
using Chain
using DataFrameMacros
using AlgebraOfGraphics
using Statistics
using Test
using DataFrames
using ProgressMeter

using AlgorithmicCompetition:
    post_prob_high_low_given_signal,
    post_prob_high_low_given_both_signals,
    draw_price_diagnostic,
    CompetitionParameters,
    CompetitionSolution,
    construct_df_summary_dddc,
    DataDemandDigitalParams,
    DDDCHyperParameters,
    draw_price_diagnostic,
    expand_and_extract_dddc,
    reduce_dddc
using Arrow

df_summary_arrow_cache_path = "data_final/dddc_v0.0.9_data_summary.arrow"

arrow_folders = filter!(
   x -> occursin(r"SLURM_ARRAY_JOB_ID=(8419083|8422841|8447799|8539762|8539372|8549184|8561296)", x),
    readdir("data", join = true),
)
arrow_files = vcat([filter(y -> occursin(".arrow", y), readdir(x, join=true)) for x in arrow_folders]...)

# TODO: For decile demand-frequency binned data, round / group to avoid losing data...
# TODO: revert to df_summary once all summary files are tested...
is_summary_file = occursin.(("df_summary",), arrow_files)
df_summary_ = arrow_files[is_summary_file]
df_raw_ = arrow_files[.!is_summary_file]

#= Temp for rebuilding summary files from raw files...
rm.(df_summary_)
@showprogress for i in 1:length(df_raw_)
    df = DataFrame(Arrow.Table(df_raw_[i]))
    if nrow(df) > 0
        df = expand_and_extract_dddc(df)
        df_summary = construct_df_summary_dddc(df)
        Arrow.write(replace(df_raw_[i], ".arrow" => "_df_summary_rebuilt.arrow"), df_summary)
    end
end
# End Temp
=#

arrows_ = DataFrame.(Arrow.Table.(df_summary_))

for i = 1:length(arrows_)
    arrows_[i][!, "metadata"] .= df_summary_[i]
    if "signal_is_strong" in names(arrows_[i])
        arrows_[i] = select!(arrows_[i], Not(:signal_is_strong))
    end
end

df_summary = vcat(arrows_...)
df_summary = reduce_dddc(df_summary, round_parameters=true)
Arrow.write(df_summary_arrow_cache_path, df_summary)

# mkpath("data_final")
# arrow_file_name = "data_final/dddc_v0.0.8_data.arrow"
# Arrow.write(arrow_file_name, df_)

# mkpath("plots/dddc")
# df_ = DataFrame(Arrow.read(arrow_file_name))

# n_simulations_dddc = @chain df_ @subset(
#     (:weak_signal_quality_level == 1) &
#     (:frequency_high_demand == 1) &
#     (:strong_signal_quality_level == :weak_signal_quality_level)
# ) nrow()

# @test (101 * 101 * 2 * n_simulations_dddc) == nrow(df_)

# df = expand_and_extract_dddc(df_)

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

# df_summary = construct_df_summary_dddc(df_)
# @assert nrow(df_summary) == 20402
# TODO: Rereduce summary data across all runs!

# Question is how existence of low state destabilizes the high state / overall collusion and to what extent...
# Question becomes 'given signal, estimated demand state prob, which opponent do I believe I am competing against?' the low demand believing opponent or the high demand one...
# in the case where own and opponents' signals are public, the high-high signal state yields the following probability curve over high state base frequency:

strong_signal_level = 0.9
df_summary = reduce_dddc(DataFrame(Arrow.Table(df_summary_arrow_cache_path)))

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
    @subset((:strong_signal_quality_level == :weak_signal_quality_level) & (:weak_signal_quality_level == round(:weak_signal_quality_level; digits=1)))
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
    @subset((:strong_signal_quality_level == :weak_signal_quality_level) & (:weak_signal_quality_level == round(:weak_signal_quality_level; digits=1)))
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
        :profit_mean => "Average Profit"
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
        (:weak_signal_quality_level == round(:weak_signal_quality_level; digits=1))
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
    @subset((:strong_signal_quality_level == :weak_signal_quality_level) & (:weak_signal_quality_level == round(:weak_signal_quality_level; digits=1)))
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
    @transform(:demand_level = replace(:demand_level, "convergence_profit_demand_" => "Demand: "))
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
    legend = (position = :top, titleposition = :left, framevisible = true, padding = 5),
)
save("plots/dddc/plot_222.svg", f222)

plt221 = @chain df_summary begin
    @subset((:strong_signal_quality_level == :weak_signal_quality_level) & (:weak_signal_quality_level == round(:weak_signal_quality_level; digits=1)))
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
    @subset((:strong_signal_quality_level == :weak_signal_quality_level) & (:weak_signal_quality_level == round(:weak_signal_quality_level; digits=1)))
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
        xticks = 0.0:0.2:1,
        yticks = 0:0.2:1.2,
        aspect = 0.5,
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
        :weak_signal_quality_level = round(:weak_signal_quality_level; digits=1),
        :signal_type = replace(
            :profit_gain_type,
            r"profit_gain_demand_[a-z]+_([a-z_]+)_signal_player" => s"\1",
        )
    )
    @subset(!ismissing(:profit_gain)) # TODO: Remove this once you figure out why missings are in data (or whether they are even in data for fresh runs...)
    @groupby(:demand_level, :weak_signal_quality_level, :strong_signal_quality_level, :signal_type, :frequency_high_demand)
    @combine(
        :profit_gain = mean(:profit_gain),
    )
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
    @subset((:frequency_high_demand != 1) & (:weak_signal_quality_level == round(:weak_signal_quality_level; digits=1)) & (:strong_signal_quality_level == strong_signal_level))
    @sort(:frequency_high_demand)
    @transform(
        :convergence_profit_type =
            replace(:convergence_profit_type, "convergence_profit_" => "")
    )
    @transform(:convergence_profit_type = replace(:convergence_profit_type, "_" => " "))
    @subset(!ismissing(:convergence_profit)) # TODO: Remove this once you figure out why missings are in data (or whether they are even in data for fresh runs...)    
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
    @subset((:weak_signal_quality_level == round(:weak_signal_quality_level; digits=1)) & (:strong_signal_quality_level == strong_signal_level))
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
    @subset((:frequency_high_demand != 1) & (:weak_signal_quality_level == round(:weak_signal_quality_level; digits=1)) & (:strong_signal_quality_level == strong_signal_level))
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
    @subset((:strong_signal_quality_level == :weak_signal_quality_level) & (:frequency_high_demand < 1.0) & (:weak_signal_quality_level == round(:weak_signal_quality_level; digits=1)))
    @subset(!ismissing(:pct_compensating_profit_gain)) # TODO: Remove this once you figure out why missings are in data (or whether they are even in data for fresh runs...)
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
            :strong_signal_quality_level == :weak_signal_quality_level ? "Weak-Weak" : "Strong-Weak"
    )
    @subset((:weak_signal_quality_level == round(:weak_signal_quality_level; digits=1)) & (:strong_signal_quality_level == strong_signal_level))
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

freq_high_demand = 0.7
for freq_high_demand in 0.0:0.1:1
    df_summary_weak_weak = @chain df_summary begin
        @subset(:frequency_high_demand == freq_high_demand)
        @subset(:weak_signal_quality_level == :strong_signal_quality_level)
        @select(:signal_quality_level = :weak_signal_quality_level, :profit_gain_avg = (:profit_gain_max + :profit_gain_min) / 2) # (no symmantic effect, but double the sample size)
    end

    plt8 = @chain df_summary begin
        @subset(:strong_signal_quality_level != 1) # TODO: remove this...
        @subset(
            # !ismissing(:profit_gain_strong_signal_player) &
            (:frequency_high_demand == freq_high_demand)
        )
        leftjoin(df_summary_weak_weak, on = :strong_signal_quality_level => :signal_quality_level, renamecols = "" => "_signal_ceil")
        leftjoin(df_summary_weak_weak, on = :weak_signal_quality_level => :signal_quality_level, renamecols = "" => "_signal_floor")
        @transform(
            :profit_gain_delta_strong_player_signal_ceil = :profit_gain_strong_signal_player - :profit_gain_avg_signal_ceil,
            :profit_gain_delta_strong_player_signal_floor = :profit_gain_strong_signal_player - :profit_gain_avg_signal_floor,
        )
        @transform(
            :profit_gain_delta_weak_player_signal_ceil = :profit_gain_weak_signal_player - :profit_gain_avg_signal_ceil,
            :profit_gain_delta_weak_player_signal_floor = :profit_gain_weak_signal_player - :profit_gain_avg_signal_floor,
        )
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
        @transform(:signal_intervention = replace(:signal_intervention, r"profit_gain_delta_.*_player_" => ""))
        @subset(!ismissing(:profit_gain_delta)) # TODO: Remove this once you figure out why missings are in data (or whether they are even in data for fresh runs...)
        data(_) *
        mapping(
            :weak_signal_quality_level,
            :strong_signal_quality_level,
            :profit_gain_delta,
            col = :signal_intervention,
            row = :player,
        ) *
        visual(Heatmap)
    end
    f8 = draw(plt8) #, axis = (xticks = 0.5:0.1:1,))
    save("plots/dddc/plot_8__freq_high_demand_$freq_high_demand.svg", f8)
end

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
    weak_signal_quality_level = 0.99,
    strong_signal_quality_level = 0.995,
    signal_is_strong = [true, false],
    frequency_high_demand = 0.9,
)

hyperparams = DDDCHyperParameters(
    α,
    β,
    δ,
    max_iter,
    competition_solution_dict,
    data_demand_digital_params;
    convergence_threshold = Int(1e5),
)

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
