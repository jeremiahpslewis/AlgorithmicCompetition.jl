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
        competition_solution::CompetitionSolution;
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

    competition_params::CompetitionParameters

    p_Bert_nash_equilibrium::Float64
    p_monop_opt::Float64

    function AIAPCHyperParameters(
        α::Float64,
        β::Float64,
        δ::Float64,
        max_iter::Int,
        competition_solution::CompetitionSolution;
        convergence_threshold::Int = Int(1e5),
        activate_extension::Bool = false, # Whether to activate the Data/Demand/Digital extension
    )
        @assert max_iter > convergence_threshold
        ξ = 0.1
        δ = 0.95
        n_prices = 15
        n_players = 2
        memory_length = 1

        # p_monop defined above
        p_range_pad =
            ξ * (
                competition_solution.p_monop_opt -
                competition_solution.p_Bert_nash_equilibrium
            )
        price_options = [
            range(
                competition_solution.p_Bert_nash_equilibrium - p_range_pad,
                competition_solution.p_monop_opt + p_range_pad,
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
            competition_solution.params,
            competition_solution.p_Bert_nash_equilibrium,
            competition_solution.p_monop_opt,
        )
    end
end


function construct_action_space(price_index, activate_extension::Bool)
    if activate_extension
        Tuple(CartesianIndex{3}(i, j, k) for i in price_index for j in price_index for k in 1:2)
    else
        Tuple(CartesianIndex{2}(i, j) for i in price_index for j in price_index)
    end
end

function initialize_memory(price_index, n_players::Int, activate_extension::Bool, high_demand_state::Bool)
    if activate_extension
        Vector{CartesianIndex}([
            CartesianIndex{3}(rand(price_index, n_players)..., high_demand_state),
        ])
    else
        Vector{CartesianIndex}([
            CartesianIndex{2}(rand(price_index, n_players)...)
        ])
    end
end
