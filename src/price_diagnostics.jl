using Test
using JuMP
using Chain
using ReinforcementLearningCore:
    PostActStage,
    PreActStage,
    state,
    reward,
    current_player,
    action_space,
    EpsilonGreedyExplorer,
    RandomPolicy,
    MultiAgentPolicy,
    RLCore,
    ResetAtTerminal
using ReinforcementLearningBase: RLBase, test_interfaces!, test_runnable!, AbstractPolicy
import ReinforcementLearningCore
using Statistics
using AlgorithmicCompetition:
    AlgorithmicCompetition,
    CompetitionParameters,
    DDDCHyperParameters,
    CompetitionParameters,
    AIAPCHyperParameters,
    AIAPCPolicy,
    AIAPCEnv,
    CompetitionSolution,
    ConvergenceCheck,
    DataDemandDigitalParams,
    solve_monopolist,
    solve_bertrand,
    p_BR,
    construct_AIAPC_state_space_lookup,
    construct_AIAPC_profit_array,
    Q,
    run,
    run_and_extract,
    Experiment,
    reward,
    InitMatrix,
    get_ϵ,
    AIAPCEpsilonGreedyExplorer,
    AIAPCSummary,
    TDLearner,
    TabularApproximator,
    economic_summary,
    extract_sim_results,
    extract_profit_vars,
    profit_gain,
    DDDCEnv,
    π
using JET
using BenchmarkTools
# using ProfileView
using Distributed

# RLCore.TimerOutputs.enable_debug_timings(RLCore)

using AlgebraOfGraphics
using CairoMakie
using DataFrames
using Chain
using DataFrameMacros

max_profit_for_price(price::Float64, price_options::Vector{Float64}, competition_params::CompetitionParameters) = maximum(first.(π.(price_options, (price,), (competition_params,))))
max_profit_for_price(price_options::Vector{Float64}, competition_params::CompetitionParameters) = max_profit_for_price.(price_options, (price_options,), (competition_params,))

min_profit_for_price(price::Float64, price_options::Vector{Float64}, competition_params::CompetitionParameters) = minimum(first.(π.(price_options, (price,), (competition_params,))))
min_profit_for_price(price_options::Vector{Float64}, competition_params::CompetitionParameters) = min_profit_for_price.(price_options, (price_options,), (competition_params,))

symmetric_profit(price::Float64, competition_params::CompetitionParameters) = first(π(price, price, competition_params))
symmetric_profit(price_options::Vector{Float64}, competition_params::CompetitionParameters) = symmetric_profit.(price_options, (competition_params,))

function extract_profit_results(profit_results, price_options)
    profit_results[:price_options] = price_options
    profit_df = @chain profit_results begin
        DataFrame
        stack(Not(:price_options), variable_name=:demand, value_name=:profit)
    end
    return profit_df
end

function generate_profit_df(hyperparams::HyperParameters, profit_for_price_function, label) where {HyperParameters <: Union{AIAPCHyperParameters, DDDCHyperParameters}}
    profit_df = Dict(demand => profit_for_price_function(hyperparams.price_options, hyperparams.competition_params_dict[demand]) for demand in [:low, :high])
    profit_df = extract_profit_results(profit_df, price_options)
    profit_df[!, :label] .= label
    return profit_df
end

function generate_profit_df(hyperparams::HyperParameters) where {HyperParameters <: Union{AIAPCHyperParameters, DDDCHyperParameters}}
    profit_df_ = [generate_profit_df(hyperparams, max_profit_for_price, "max_profit"),
        generate_profit_df(hyperparams, min_profit_for_price, "min_profit"),
        generate_profit_df(hyperparams, symmetric_profit, "symmetric_profit")]
    profit_df = vcat(profit_df_...)
    return profit_df
end



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
    low_signal_quality_level = 0.99,
    high_signal_quality_level = 0.995,
    signal_quality_is_high = [true, false],
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

profit_df = generate_profit_df(hyperparams)
profit_df = unstack(profit_df, :label, :profit)

critical_prices = vcat([[hyperparams.p_Bert_nash_equilibrium[demand], hyperparams.p_monop_opt[demand]] for demand in [:high, :low]]...)
critical_profits = vcat([symmetric_profit([hyperparams.p_Bert_nash_equilibrium[demand], hyperparams.p_monop_opt[demand]], hyperparams.competition_params_dict[demand]) for demand in [:high, :low]]...)
plt_1 = data(
    (
        price = critical_prices,
        profit = critical_profits,
        label = repeat(["Bertrand Nash", "Monopoly"], outer=2),
        demand = repeat(["high", "low"], inner=2),
    )
    ) *
    mapping(
        :price,
        :profit,
        color = :label,
        row = :demand,
    ) *
    visual(Scatter)

plt = data(profit_df) *
    mapping(
        :price_options => "Price",
        :symmetric_profit => "Profit",
        lower = :min_profit,
        upper = :max_profit,
        row = :demand => "Demand Level",
    ) *
    (visual(Scatter) + visual(LinesFill))
draw(plt + plt_1, axis=(title="Profit Levels across Price Options", subtitle="(Solid line is profit for symmetric prices, shaded region shows range based on competitor prices)", xlabel="Competitor's Price Choice",))


α = Float64(0.125)
β = Float64(4e-1)
δ = 0.95
ξ = 0.1
δ = 0.95
n_prices = 15
max_iter = Int(1e6)
price_index = 1:n_prices

competition_params_dict = Dict(
    :high => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
    :low => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
)

competition_solution_dict =
    Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])


hyperparams = AIAPCHyperParameters(
    α,
    β,
    δ,
    max_iter,
    competition_solution_dict;
    convergence_threshold = Int(1e5),
)

profit_df = generate_profit_df(hyperparams)
profit_df = unstack(profit_df, :label, :profit)

critical_prices = [hyperparams.p_Bert_nash_equilibrium, hyperparams.p_monop_opt]
plt_1 = data((price = critical_prices, profit = symmetric_profit(critical_prices, hyperparams.competition_params_dict[:high]), label = ["Bertrand Nash", "Monopoly"])) *
    mapping(
        :price,
        :profit,
        color = :label,
    ) *
    visual(Scatter)

plt = data(profit_df) *
    mapping(
        :price_options => "Price",
        :symmetric_profit => "Profit",
        lower = :min_profit,
        upper = :max_profit,
        color = :demand => "Demand Level",
    ) *
    (visual(Scatter) + visual(LinesFill))
draw(plt + plt_1, axis=(title="Profit Levels across Price Options", subtitle="(Solid line is profit for symmetric prices, shaded region shows range based on competitor prices)",))


