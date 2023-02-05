struct CompetitionSolution
    p_Bert_nash_equilibrium::Float64
    p_monop_opt::Float64
    profit_function::Any

    function CompetitionSolution(params::CompetitionParameters)
        model_monop, p_monop = solve_monopolist(params)

        p_Bert_nash_equilibrium = solve_bertrand(params)[2][1]
        p_monop_opt = solve_monopolist(params)[2][1]

        profit_function = p -> π_fun(p, params)

        new(p_Bert_nash_equilibrium, p_monop_opt, profit_function)
    end
end

struct CalvanoHyperParameters
    α::Float64
    β::Float64
    δ::Float64
    price_options::Vector{Float64}
    memory_length::Int64
    n_players::Int64
    max_iter::Int64
    convergence_threshold::Int64
    profit_function::Any
    p_Bert_nash_equilibrium::Float64
    p_monop_opt::Float64

    function CalvanoHyperParameters(
        α::Float64,
        β::Float64,
        δ::Float64,
        max_iter::Int64,
        competition_solution::CompetitionSolution,
    )
        ξ = 0.1
        δ = 0.95
        n_prices = 15
        n_players = 2
        memory_length = 2

        # p_monop defined above
        p_range_pad =
            ξ * (
                competition_solution.p_monop_opt -
                competition_solution.p_Bert_nash_equilibrium
            )
        price_options = [
            range(
                competition_solution.p_Bert_nash_equilibrium,
                competition_solution.p_monop_opt,
                n_prices,
            )...,
        ]

        convergence_threshold = Int(1e5)

        new(
            α,
            β,
            δ,
            price_options,
            memory_length,
            n_players,
            max_iter,
            convergence_threshold,
            competition_solution.profit_function,
            competition_solution.p_Bert_nash_equilibrium,
            competition_solution.p_monop_opt,
        )
    end
end