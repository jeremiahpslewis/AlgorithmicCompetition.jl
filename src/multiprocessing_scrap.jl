using Distributed
# using ParallelDataTransfer

_procs = addprocs(4, topology=:master_worker, exeflags=["--threads=1", "--project=$(Base.active_project())"])

@everywhere begin
    using Pkg; Pkg.instantiate()
    using AlgorithmicCompetition   
end

const α = 0.125
const β = 1e-5
const δ = 0.95
const ξ = 0.1
const δ = 0.95
const n_prices = 15
const price_index = 1:n_prices

const competition_params = AlgorithmicCompetition.CompetitionParameters(
        μ = 0.25,
        a_0 = 0,
        a = [2, 2],
        c = [1, 1],
        n_firms = 2,
    )

model_monop, p_monop = AlgorithmicCompetition.solve_monopolist(competition_params)

const p_Bert_nash_equilibrium = AlgorithmicCompetition.solve_bertrand(competition_params)[2][1]
const p_monop_opt = AlgorithmicCompetition.solve_monopolist(competition_params)[2][1]

# p_monop defined above
p_range_pad = ξ * (p_monop_opt - p_Bert_nash_equilibrium)
const price_options = [range(p_Bert_nash_equilibrium, p_monop_opt, n_prices)...]


n_increments = 2 # for full sim, use 100
α_ = range(0.025, 0.25, n_increments)
# TODO: Fix this parameterization based on Calvano pg. 12
β_ = range(1.25e-8, 2e-5, n_increments) 

const param_set = [
    (α, β, δ,
     price_options,
     competition_params,
     p_Bert_nash_equilibrium,
     p_monop_opt
    ) for α in α_ for β in β_
    ]

# ν function from Calvano pg. 12
ν(β, k, m, n) = (m - 1)^n / ((m^(k * n * (n + 1))) * (1 - exp(- β * (n+1))))
ν(β, 1, 15, 2) # n = n_firms, m = n_prices, k = memory(?)
ν(1.25e-8, 1, 15, 2) # Value for ≈ 450

# Transfer Data to Workers
# sendto(workers(), δ=δ,
#     param_set=param_set,
#     price_options=price_options,
#     competition_params=competition_params,
#     p_Bert_nash_equilibrium=p_Bert_nash_equilibrium,
#     p_monop_opt=p_monop_opt,
#     )

# function pmap_experiments()
#     return pmap(param_ -> AlgorithmicCompetition.run_pmap(
#         param_[1],
#         param_[2],
#         δ,
#         price_options,
#         competition_params,
#         p_Bert_nash_equilibrium,
#         p_monop_opt,
#         ), param_set)
# end

# e_out = pmap_experiments()
# rmprocs(_procs)

a = pmap(AlgorithmicCompetition.run_pmap, _procs, param_set)

# TODOs
# - Add hook to save profit results
# - Get multiprocessing loop running
# - Run and save results


# init_matrix = hcat(fill([Q_0(price_options, δ)...], n_state_space)...)
# TODO: Parametrize init matrix, use above matrix as default

