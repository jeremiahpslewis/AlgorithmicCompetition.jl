using AlgorithmicCompetition
using Chain

α = 0.125
β = 1e-5
δ = 0.95
ξ = 0.1
δ = 0.95
n_prices = 15
price_index = 1:n_prices

competition_params = AlgorithmicCompetition.CompetitionParameters(
        μ = 0.25,
        a_0 = 0,
        a = [2, 2],
        c = [1, 1],
        n_firms = 2
    )

model_monop, p_monop = AlgorithmicCompetition.solve_monopolist(competition_params)

p_Bert_nash_equilibrium = AlgorithmicCompetition.solve_bertrand(competition_params)[2][1]
p_monop_opt = AlgorithmicCompetition.solve_monopolist(competition_params)[2][1]

# p_monop defined above
p_range_pad = ξ * (p_monop_opt - p_Bert_nash_equilibrium)
price_options = [range(p_Bert_nash_equilibrium, p_monop_opt, n_prices)...]

# AlgorithmicCompetition.runCalvano(α, β, δ, price_options, competition_params, max_iter=1000)


# env = AlgorithmicCompetition.CalvanoEnv(
#     α,
#     β,
#     δ,
#     2, # n_players
#     2, # memory_length
#     price_options,
#     100,
#     10,
#     (p_1, p_2) -> p_1 + p_2,
# )

# AlgorithmicCompetition.CalvanoEnv(; α, β, δ, n_players, memory_length, price_options, max_iter, convergence_threshold, profit_function)

using JET
using BenchmarkTools
# @time AlgorithmicCompetition.runCalvano(α, β, δ, price_options, competition_params)
@time AlgorithmicCompetition.runCalvano(α, β, δ, price_options, competition_params, max_iter=1000)
