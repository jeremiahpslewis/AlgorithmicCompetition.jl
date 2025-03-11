using CairoMakie
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
using Tidier

df_summary_arrow_cache_path = "data/SLURM_ARRAY_JOB_ID=0_debug=true_model=dddc_version=v0.1.1/SLURM_ARRAY_TASK_ID=0_start_timestamp=2025-03-11T14:11:10.744_df_summary.arrow"
df_full = DataFrame(Arrow.Table(df_summary_arrow_cache_path))
df_summary = AlgorithmicCompetition.reduce_dddc(df_full)

mkpath("plots/acai")

demand_cat(x) = x == 1 ? "Always High" : x == 0 ? "Always Low" : x == 0.5 ? "High / Low Split" : "Invalid"
function signal_cat(signal_quality_level)
    if signal_quality_level == 1
        return "Perfect"
    elseif signal_quality_level == 0.5
        return "Noise"
    elseif signal_quality_level == 0
        return "None"
    else
        error("Invalid signal quality level: $signal_quality_level")
    end
end

v1 = @chain df_summary begin
    @filter((weak_signal_quality_level ∈ [0.5, 1.0]) & (strong_signal_quality_level ∈ [0.5, 1.0]) & (strong_signal_quality_level == weak_signal_quality_level))
    @mutate(
        signal_quality_level = categorical(signal_cat(weak_signal_quality_level), levels=["None", "Perfect", "Noise"]),
        profit_gain = (profit_gain_min + profit_gain_max) / 2,
        demand_scenario = demand_cat(frequency_high_demand)
    )
    @select(signal_quality_level, demand_scenario, profit_gain)
    data(_) *
    mapping(
        :signal_quality_level => nonnumeric => "",
        :profit_gain => "Profit Gain",
        color = :signal_quality_level => nonnumeric => "Scenario",
        col = :demand_scenario => nonnumeric => "Demand Scenario",
    ) *
    (visual(BarPlot))
end

f1 = draw(v1)
save("plots/acai/plot_1_barplot_profit_gain_by_signal_and_demand_scenario.svg", f1)

v1 = @chain df_summary begin
    @filter((weak_signal_quality_level ∈ [0.0, 0.5, 1.0]) & (strong_signal_quality_level ∈ [0.0, 0.5, 1.0]) & (strong_signal_quality_level == weak_signal_quality_level))
    @mutate(
        signal_quality_level = categorical(signal_cat(weak_signal_quality_level), levels=["None", "Perfect", "Noise"], ordered=true),
        profit_gain = (profit_gain_min + profit_gain_max) / 2,
        demand_scenario = demand_cat(frequency_high_demand)
    )
    @select(signal_quality_level, demand_scenario, profit_gain)
    data(_) *
    mapping(
        :signal_quality_level => nonnumeric => "",
        :profit_gain => "Profit Gain",
        color = :signal_quality_level => nonnumeric => "Scenario",
        col = :demand_scenario => nonnumeric => "Demand Scenario",
    ) *
    (visual(BarPlot))
end

f1 = draw(v1)
save("plots/acai/plot_1_barplot_profit_gain_by_signal_and_demand_scenario.svg", f1)
