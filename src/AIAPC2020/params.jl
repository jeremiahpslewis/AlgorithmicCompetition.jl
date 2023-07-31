"""
    CompetitionSolution(params::CompetitionParameters)

Solve the monopolist and Bertrand competition models for the given parameters and return the solution.
"""
struct CompetitionSolution
    p_Bert_nash_equilibrium::Float64
    p_monop_opt::Float64
    params::CompetitionParameters

    function CompetitionSolution(params::CompetitionParameters)
        model_monop, p_monop = solve_monopolist(params)

        p_Bert_nash_equilibrium = solve_bertrand(params)[2][1]
        p_monop_opt = solve_monopolist(params)[2][1]

        new(p_Bert_nash_equilibrium, p_monop_opt, params)
    end
end

"""
    AIAPCHyperParameters(
        α::Float64,
        β::Float64,
        δ::Float64,
        max_iter::Int,
        competition_solution_dict::Dict{Symbol,CompetitionSolution};
        convergence_threshold::Int = Int(1e5),
    )

Hyperparameters which define a specific AIAPC environment.
"""
struct AIAPCHyperParameters
    α::Float64
    β::Float64
    δ::Float64
    max_iter::Int
    convergence_threshold::Int

    price_options::Vector{Float64}
    memory_length::Int
    n_players::Int

    competition_params_dict::Dict{Symbol,CompetitionParameters}

    p_Bert_nash_equilibrium::Float64
    p_monop_opt::Float64

    demand_mode::Symbol

    function AIAPCHyperParameters(
        α::Float64,
        β::Float64,
        δ::Float64,
        max_iter::Int,
        competition_solution_dict::Dict{Symbol,CompetitionSolution};
        convergence_threshold::Int = Int(1e5),
        demand_mode::Symbol = :high,
    )
        @assert max_iter > convergence_threshold
        @assert demand_mode ∈ [:high, :low]
        ξ = 0.1
        δ = 0.95
        n_prices = 15
        n_players = 2
        memory_length = 1

        # p_monop defined above
        p_range_pad =
            ξ * (
                competition_solution_dict[demand_mode].p_monop_opt -
                competition_solution_dict[demand_mode].p_Bert_nash_equilibrium
            )
        price_options = [
            range(
                competition_solution_dict[demand_mode].p_Bert_nash_equilibrium -
                p_range_pad,
                competition_solution_dict[demand_mode].p_monop_opt + p_range_pad,
                n_prices,
            )...,
        ]

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
            competition_solution_dict[demand_mode].p_Bert_nash_equilibrium,
            competition_solution_dict[demand_mode].p_monop_opt,
            demand_mode,
        )
    end
end


function construct_AIAPC_action_space(price_index)
    Tuple(CartesianIndex{2}(i, j) for i in price_index for j in price_index)
end

function initialize_price_memory(price_index, n_players::Int)
    Vector{CartesianIndex}([CartesianIndex{2}(rand(price_index, n_players)...)])
end
