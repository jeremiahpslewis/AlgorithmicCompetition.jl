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

arrow_files = readdir("data/SLURM_ARRAY_JOB_ID=0_debug=false_model=dddc_version=v0.1.2", join = true)
arrow_files = filter(y -> occursin("df_summary.arrow", y), arrow_files)
df_full = vcat(DataFrame.(Arrow.Table.(arrow_files))...)
df_summary = AlgorithmicCompetition.reduce_dddc(df_full)

mkpath("plots/acai")

demand_cat(x) = x == 1 ? "Always High" : x == 0 ? "Always Low" : x == 0.5 ? "High / Low Split" : "Invalid"
function signal_cat(weak_signal_quality_level, strong_signal_quality_level)
    if weak_signal_quality_level == strong_signal_quality_level
        if weak_signal_quality_level == 1
            return "Perfect"
        elseif weak_signal_quality_level == 0.5
            return "Random"
        elseif weak_signal_quality_level == 0
            return "None"
        elseif weak_signal_quality_level == -1
            return "Common Sunspot"
        end
    else
        if strong_signal_quality_level == 1 && weak_signal_quality_level == 0.5
            return "Perfect / Random Split"
        end
    end

    error("Invalid signal quality level: $weak_signal_quality_level, $strong_signal_quality_level")
end

edge_cases = [0.5, 1.0, 0.0, -1.0]
key_viz_data = @chain df_summary begin
    @filter((weak_signal_quality_level ∈ !!edge_cases) & (strong_signal_quality_level ∈ !!edge_cases))
    @mutate(
        signal_quality_level = categorical(signal_cat(weak_signal_quality_level, strong_signal_quality_level), levels=["None", "Perfect", "Common Sunspot", "Perfect / Random Split", "Random"], ordered=true),
        profit_gain = (profit_gain_min + profit_gain_max) / 2,
        demand_scenario = demand_cat(frequency_high_demand)
    )
    @select(signal_quality_level, demand_scenario, profit_gain, profit_mean)
end

v1 = @chain key_viz_data begin
    data(_) *
    mapping(
        :signal_quality_level => nonnumeric => "",
        :profit_gain => "Profit Gain",
        color = :signal_quality_level => nonnumeric => "Demand Signal",
        col = :demand_scenario => nonnumeric => "Demand Environment",
    ) *
    (visual(BarPlot))
end

f1 = draw(v1, axis = (; xticklabelrotation = 45),
figure = (; size = (800, 600), title = "Algorithmic Collusion Outcomes by Information Set", subtitle="Mean of $(df_summary[1, :n_obs]) simulations per scenario", fontsize = 16))
save("plots/acai/plot_1_barplot_profit_gain_by_signal_and_demand_scenario.svg", f1)

v2 = @chain key_viz_data begin
    data(_) *
    mapping(
        :signal_quality_level => nonnumeric => "",
        :profit_mean => "Avg. Profit",
        color = :signal_quality_level => nonnumeric => "Demand Signal",
        col = :demand_scenario => nonnumeric => "Demand Environment",
    ) *
    (visual(BarPlot))
end

f2 = draw(v2, axis = (; xticklabelrotation = 45),
figure = (; size = (800, 600), title = "Algorithmic Collusion Outcomes by Information Set", subtitle="Mean of $(df_summary[1, :n_obs]) simulations per scenario", fontsize = 16))
save("plots/acai/plot_2_barplot_avg_profit_by_signal_and_demand_scenario.svg", f2)
