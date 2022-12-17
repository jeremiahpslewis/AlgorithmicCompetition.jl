using Revise
using AlgorithmicCompetition
using Chain

const α = 0.125
const β = 1e-5
const δ = 0.95
const ξ = 0.1
const δ = 0.95
const n_prices = 15
const price_index = 1:n_prices

competition_params = AlgorithmicCompetition.CompetitionParameters(
        μ = 0.25,
        a_0 = 0,
        a = [2, 2],
        c = [1, 1],
        n_firms = 2,
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
@btime AlgorithmicCompetition.runCalvano(α, β, δ, price_options, competition_params, max_iter=1000)
# @report_opt AlgorithmicCompetition.runCalvano(α, β, δ, price_options, competition_params)#, max_iter=1000)


# 15.775 ms (186059 allocations: 16.66 MiB)
# 15.430 ms (186076 allocations: 16.66 MiB)
# 15.072 ms (185833 allocations: 16.63 MiB)
# 18.628 ms (208519 allocations: 16.97 MiB)
# 15.904 ms (185829 allocations: 16.63 MiB) # UInt8 seems to help!
# 15.725 ms (185835 allocations: 16.63 MiB)
# 15.130 ms (185794 allocations: 15.40 MiB) # UInt8
# 15.456 ms (185787 allocations: 12.41 MiB)
# 15.283 ms (185782 allocations: 12.41 MiB)
# TODO:
# - [ ] Figure out why n_state_space is 5k not ~200
