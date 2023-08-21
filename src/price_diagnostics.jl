using AlgebraOfGraphics
using CairoMakie
using DataFrames
using Chain
using DataFrameMacros

max_profit_for_price(
    price::Float64,
    price_options::Vector{Float64},
    competition_params::CompetitionParameters,
) = maximum(first.(π.(price_options, (price,), (competition_params,))))
max_profit_for_price(
    price_options::Vector{Float64},
    competition_params::CompetitionParameters,
) = max_profit_for_price.(price_options, (price_options,), (competition_params,))

min_profit_for_price(
    price::Float64,
    price_options::Vector{Float64},
    competition_params::CompetitionParameters,
) = minimum(first.(π.(price_options, (price,), (competition_params,))))
min_profit_for_price(
    price_options::Vector{Float64},
    competition_params::CompetitionParameters,
) = min_profit_for_price.(price_options, (price_options,), (competition_params,))

symmetric_profit(price::Float64, competition_params::CompetitionParameters) =
    first(π(price, price, competition_params))
symmetric_profit(
    price_options::Vector{Float64},
    competition_params::CompetitionParameters,
) = symmetric_profit.(price_options, (competition_params,))

function extract_profit_results(profit_results, price_options)
    profit_results[:price_options] = price_options
    profit_df = @chain profit_results begin
        DataFrame
        stack(Not(:price_options), variable_name = :demand, value_name = :profit)
    end
    return profit_df
end

function generate_profit_df(
    hyperparams::HyperParameters,
    profit_for_price_function,
    label,
) where {HyperParameters<:Union{AIAPCHyperParameters,DDDCHyperParameters}}
    profit_df = Dict(
        demand => profit_for_price_function(
            hyperparams.price_options,
            hyperparams.competition_params_dict[demand],
        ) for demand in [:low, :high]
    )
    profit_df = extract_profit_results(profit_df, hyperparams.price_options)
    profit_df[!, :label] .= label
    return profit_df
end

function generate_profit_df(
    hyperparams::HyperParameters,
) where {HyperParameters<:Union{AIAPCHyperParameters,DDDCHyperParameters}}
    profit_df_ = [
        generate_profit_df(hyperparams, max_profit_for_price, "max_profit"),
        generate_profit_df(hyperparams, min_profit_for_price, "min_profit"),
        generate_profit_df(hyperparams, symmetric_profit, "symmetric_profit"),
    ]
    profit_df = vcat(profit_df_...)
    return profit_df
end

function draw_price_diagnostic(hyperparams::AIAPCHyperParameters)
    profit_df = generate_profit_df(hyperparams)
    profit_df = unstack(profit_df, :label, :profit)

    critical_prices = [hyperparams.p_Bert_nash_equilibrium, hyperparams.p_monop_opt]
    plt_1 =
        data((
            price = critical_prices,
            profit = symmetric_profit(
                critical_prices,
                hyperparams.competition_params_dict[:high],
            ),
            label = ["Bertrand Nash", "Monopoly"],
        )) *
        mapping(:price, :profit, color = :label) *
        visual(Scatter)

    plt = @chain profit_df begin
        @subset(:demand == "high")
        data(_) *
        mapping(
            :price_options => "Price",
            :symmetric_profit => "Profit",
            lower = :min_profit,
            upper = :max_profit,
        ) *
        (visual(Scatter) + visual(LinesFill))
    end
    return plt + plt_1
end

function draw_price_diagnostic(hyperparams::DDDCHyperParameters)
    profit_df = generate_profit_df(hyperparams)
    profit_df = unstack(profit_df, :label, :profit)

    critical_prices = vcat(
        [
            [hyperparams.p_Bert_nash_equilibrium[demand], hyperparams.p_monop_opt[demand]] for demand in [:high, :low]
        ]...,
    )
    critical_profits = vcat(
        [
            symmetric_profit(
                [
                    hyperparams.p_Bert_nash_equilibrium[demand],
                    hyperparams.p_monop_opt[demand],
                ],
                hyperparams.competition_params_dict[demand],
            ) for demand in [:high, :low]
        ]...,
    )
    plt_1 =
        data((
            price = critical_prices,
            profit = critical_profits,
            label = repeat(["Bertrand Nash", "Monopoly"], outer = 2),
            demand = repeat(["High Demand", "Low Demand"], inner = 2),
        )) *
        mapping(
            :price,
            :profit => "Profit",
            color = :label => "Equilibria",
            row = :demand
        ) *
        visual(Scatter)

    plt =
        @chain profit_df begin
        @transform(:demand = :demand == "high" ? "High Demand" : "Low Demand")
        data(_) *
        mapping(
            :price_options => "Price",
            :symmetric_profit => "Profit",
            lower = :min_profit,
            upper = :max_profit,
            row = :demand => "Demand Level",
        ) *
        (visual(Scatter) + visual(LinesFill))
    end
    return plt + plt_1
end
