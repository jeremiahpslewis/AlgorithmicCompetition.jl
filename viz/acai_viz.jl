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
    DDDCExperimentalParams,
    DDDCHyperParameters
using AlgorithmicCompetition
using Arrow
using Tidier

use_summary_files = true

arrow_files = readdir(
    "data/SLURM_ARRAY_JOB_ID=83335_debug=false_model=dddc_version=2025-03-19-dddc-trembling-hand",
    join = true,
)
arrow_files = filter(y -> occursin(".arrow", y), arrow_files)

if use_summary_files
    arrow_files = filter(y -> occursin("df_summary", y), arrow_files)
    df_full = vcat(DataFrame.(Arrow.Table.(arrow_files))...)
else
    arrow_files = filter(y -> !occursin("df_summary", y), arrow_files)
    df_full_ = AlgorithmicCompetition.build_summary_from_raw_arrow_file.(arrow_files)
    df_full = vcat(df_full_...)
end

df_summary = AlgorithmicCompetition.reduce_dddc(df_full)

mkpath("plots/acai")

demand_cat(x) =
    x == 1 ? "Always High Demand" :
    x == 0 ? "Always Low Demand" : x == 0.5 ? "High / Low Split" : "Invalid"
function signal_cat(weak_signal_quality_level, strong_signal_quality_level)
    if weak_signal_quality_level == strong_signal_quality_level
        if weak_signal_quality_level == 1
            return "True State"
        elseif weak_signal_quality_level == 0.5
            return "Independent Random"
        elseif weak_signal_quality_level == 0
            return "No Signal"
        elseif weak_signal_quality_level == -1
            return "Common Random"
        end
    elseif strong_signal_quality_level == 1 && weak_signal_quality_level == 0.5
        return "P1 Perfect / P2 Random"
    end

    error(
        "Invalid signal quality level: $weak_signal_quality_level, $strong_signal_quality_level",
    )
end

edge_cases = [0.5, 1.0, 0.0, -1.0]
key_viz_data = @chain df_summary begin
    @filter(
        (weak_signal_quality_level ∈ !!edge_cases) &
        (strong_signal_quality_level ∈ !!edge_cases)
    )
    @filter(
        !(
            (weak_signal_quality_level == 1) &
            (strong_signal_quality_level == 1) &
            (frequency_high_demand ∈ [0, 1])
        )
    )
    @mutate(
        signal_quality_level = categorical(
            signal_cat(weak_signal_quality_level, strong_signal_quality_level),
            levels = [
                "No Signal",
                "True State",
                "Common Random",
                "P1 Perfect / P2 Random",
                "Independent Random",
            ],
            ordered = true,
        ),
        profit_gain = (profit_gain_min + profit_gain_max) / 2,
        demand_scenario = demand_cat(frequency_high_demand)
    )
    # @filter(signal_quality_level != "P1 Perfect / P2 Random")
    # @filter(signal_quality_level != "True State") # Might be interesting to look at signal-conditional memory, e.g. remember prices and state from last x periods in which signal was same as current...
    @select(
        signal_quality_level,
        demand_scenario,
        profit_gain,
        profit_mean,
        trembling_hand_frequency
    )
end

v1 = @chain key_viz_data begin
    data(_) *
    mapping(
        :signal_quality_level => nonnumeric => "",
        :profit_gain => "Profit Gain",
        color = :signal_quality_level => nonnumeric => "Demand Signal",
        col = :demand_scenario => nonnumeric => "Demand Environment",
        row = :trembling_hand_frequency => nonnumeric => "Trembling Hand Frequency",
    ) *
    (visual(BarPlot))
end

f1 = draw(
    v1,
    axis = (; xticklabelrotation = 45),
    figure = (;
        size = (800, 1000),
        title = "Algorithmic Collusion Outcomes by Information Set",
        subtitle = "Mean of $(df_summary[1, :n_obs]) simulations per scenario",
        fontsize = 16,
        xlabel = "Information Set",
    ),
)
save("plots/acai/plot_1_barplot_profit_gain_by_signal_and_demand_scenario.svg", f1)

v2 = @chain key_viz_data begin
    data(_) *
    mapping(
        :signal_quality_level => nonnumeric => "",
        :profit_mean => "Avg. Profit",
        color = :signal_quality_level => nonnumeric => "Demand Signal",
        col = :demand_scenario => nonnumeric => "Demand Environment",
        row = :trembling_hand_frequency => nonnumeric => "Trembling Hand Frequency",
    ) *
    (visual(BarPlot))
end

f2 = draw(
    v2,
    axis = (; xticklabelrotation = 45),
    figure = (;
        size = (800, 1000),
        title = "Algorithmic Collusion Outcomes by Information Set",
        subtitle = "Mean of $(df_summary[1, :n_obs]) simulations per scenario",
        fontsize = 16,
        xlabel = "Information Set",
    ),
)
save("plots/acai/plot_2_barplot_avg_profit_by_signal_and_demand_scenario.svg", f2)
