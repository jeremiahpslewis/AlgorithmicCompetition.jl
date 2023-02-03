using Test
using JuMP
using AlgorithmicCompetition: AlgorithmicCompetition, CompetitionParameters, solve_monopolist, solve_bertrand, p_BR, map_memory_to_state, runCalvano, q_fun

p_monop_opt = 1.92498
p_Bert_nash_equilibrium = 1.47

α = 0.125
β = 1e-5
δ = 0.95
ξ = 0.1
δ = 0.95
n_prices = 15
price_index = 1:n_prices

competition_params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])
p_range_pad = ξ * (p_monop_opt - p_Bert_nash_equilibrium)
price_options = [range(p_Bert_nash_equilibrium, p_monop_opt, n_prices)...]

a = runCalvano(α, β, δ,
    price_options,
    competition_params,
    p_Bert_nash_equilibrium,
    p_monop_opt,
)
