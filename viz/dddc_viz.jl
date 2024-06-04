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

arrow_folders = filter!(
   x -> occursin("SLURM_ARRAY_JOB_ID=(8419083|8422841)", x),
    readdir("data", join = true),
)
arrow_files = vcat([filter(y -> occursin(".arrow", y), readdir(x, join=true)) for x in arrow_folders]...)

is_summary_file = occursin.(("df_summary",), arrow_files)
df_summary_ = arrow_files[is_summary_file]
df_raw_ = arrow_files[.!is_summary_file]

arrows_ = DataFrame.(Arrow.Table.(df_summary_))

for i = 1:length(arrows_)
    arrows_[i][!, "metadata"] .= df_summary_[i]
end

df_summary = vcat(arrows_...)
df_summary = reduce_dddc(df_summary)
# mkpath("data_final")
# arrow_file_name = "data_final/dddc_v0.0.8_data.arrow"
# Arrow.write(arrow_file_name, df_)

# mkpath("plots/dddc")
# df_ = DataFrame(Arrow.read(arrow_file_name))

# n_simulations_dddc = @chain df_ @subset(
#     (:weak_signal_quality_level == 1) &
#     (:frequency_high_demand == 1) &
#     (:signal_is_strong == [0, 0])
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
@assert nrow(df_summary) == 20402
# TODO: Rereduce summary data across all runs!

Arrow.write("data_final/dddc_v0.0.8_data_summary.arrow", df_summary)

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
            pr_high_demand^2 * pr_signal_true,
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
            xticks = 0.5:0.1:1,
            yticks = 0:0.1:1,
            xlabel = "Probability High Demand",
            ylabel = "Probability High Demand and Opponent Signal High Given Own Signal High",
        ),
    )
end
save("plots/dddc/plot_11.svg", f11)

4
