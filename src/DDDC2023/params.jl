function construct_DDDC_action_space(price_index)
    Tuple(
        CartesianIndex{4}(i, j, k, l) for i in price_index for j in price_index for k = 1:2
        for l = 1:2
    )
end

"""
    DDDCHyperParameters(
        α::Float64,
        β::Float64,
        δ::Float64,
        max_iter::Int,
        competition_solution_dict::Dict{Symbol,CompetitionSolution},
        data_demand_digital_params::DataDemandDigitalParams;
        convergence_threshold::Int = Int(1e5),
    )

Hyperparameters which define a specific DDDC environment.
"""
struct DDDCHyperParameters
    α::Float64
    β::Float64
    δ::Float64
    max_iter::Int
    convergence_threshold::Int

    price_options::Vector{Float64}
    memory_length::Int
    n_players::Int

    competition_params_dict::Dict{Symbol,CompetitionParameters}

    p_Bert_nash_equilibrium::Dict{Symbol,Float64}
    p_monop_opt::Dict{Symbol,Float64}

    data_demand_digital_params::DataDemandDigitalParams

    function DDDCHyperParameters(
        α::Float64,
        β::Float64,
        δ::Float64,
        max_iter::Int,
        competition_solution_dict::Dict{Symbol,CompetitionSolution},
        data_demand_digital_params::DataDemandDigitalParams;
        convergence_threshold::Int = Int(1e5),
    )
        @assert max_iter > convergence_threshold
        ξ = 0.1
        δ = 0.95
        n_prices = 6
        n_players = 2
        memory_length = 1

  
        p_Bert_nash_equilibrium = [
            competition_solution_dict[demand_mode].p_Bert_nash_equilibrium for
            demand_mode in [:high, :low]
        ]
        p_monop_opt = [
            competition_solution_dict[demand_mode].p_monop_opt for
            demand_mode in [:high, :low]
        ]
        p_critical = [p_Bert_nash_equilibrium..., p_monop_opt...]
        p_range_pad = ξ * (maximum(p_critical) - minimum(p_critical))
        # Price options are high and low demand critical prices, upper and lower bounds padded
        price_options = [
                minimum(p_critical) - p_range_pad,
                maximum(p_critical) + p_range_pad,
                p_critical...
        ]
        sort!(price_options)
        @assert length(price_options) == n_prices
        new(
            α,
            β,
            δ,
            max_iter,
            convergence_threshold,
            price_options,
            memory_length,
            n_players,
            Dict(d_ => competition_solution_dict[d_].params for d_ in [:high, :low]),
            Dict(
                d_ => competition_solution_dict[d_].p_Bert_nash_equilibrium for
                d_ in [:high, :low]
            ),
            Dict(d_ => competition_solution_dict[d_].p_monop_opt for d_ in [:high, :low]),
            data_demand_digital_params,
        )
    end
end
