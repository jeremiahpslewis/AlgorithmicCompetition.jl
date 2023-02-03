using Distributed
using AlgorithmicCompetition: AlgorithmicCompetition, CalvanoHyperParameters, CalvanoEnv, Experiment
using ReinforcementLearning

multiproc = false

if multiproc
    using ParallelDataTransfer

    _procs = addprocs(1,
        topology=:master_worker,
        exeflags=["--threads=1", "--project=$(Base.active_project())"]
        )

    @everywhere begin
        using Pkg; Pkg.instantiate()
        using AlgorithmicCompetition: AlgorithmicCompetition, CalvanoHyperParameters, CalvanoEnv
    end
end

n_increments = 100
max_iter = Int(1e6) # Should be 1e9
α_ = range(0.025, 0.25, n_increments)
β_ = range(1.25e-8, 2e-5, n_increments) 
δ = 0.95

alpha_beta = [(α, β) for α in α_ for α_ in β_]

@chain alpha_beta begin
    CalvanoHyperParameters.(alpha_beta[1], alpha_beta[2], (δ,), (max_iter,))
    CalvanoEnv.()
    Experiment.()
end

params = CalvanoHyperParameters.(α_, β_, (δ,), (max_iter,))
env = CalvanoEnv.(params)
experiment = Experiment.(env)

CalvanoHyperParameters.(α_, β_, (δ,), (max_iter,)) |> CalvanoEnv.() |> Experiment.()

if multiproc
    # Transfer Data to Workers
    sendto(workers(), params=params
        )

    pmap(AlgorithmicCompetition.run, param_set)
else
    Base.run.(experiment)
end

# const α = 0.125
# const β = 1e-5
# const δ = 0.95
# const ξ = 0.1
# const δ = 0.95
# const n_prices = 15
# const price_index = 1:n_prices

# # ν function from Calvano pg. 12
# ν(β, k, m, n) = (m - 1)^n / ((m^(k * n * (n + 1))) * (1 - exp(- β * (n+1))))
# ν(β, 1, 15, 2) # n = n_firms, m = n_prices, k = memory(?)
# ν(1.25e-8, 1, 15, 2) # Value for ≈ 450

# n_increments = 2 #s for full sim, use 100

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


# TODOs
# - Add hook to save profit results
# - Get multiprocessing loop running
# - Run and save results


# init_matrix = hcat(fill([Q_0(price_options, δ)...], n_state_space)...)
# TODO: Parametrize init matrix, use above matrix as default
