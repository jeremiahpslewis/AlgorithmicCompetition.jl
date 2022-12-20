using Distributed
using ParallelDataTransfer

addprocs(7, topology=:master_worker, exeflags=["--threads=1", "--project=$(Base.active_project())"])

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


n_increments = 3 # for full sim, use 100
α_ = range(0.025, 0.25, n_increments)
# TODO: Fix this parameterization based on Calvano pg. 12
β_ = range(0, 2e-5, α = range(0.025, 0.25, n_increments) 
param_set = [(α, β) for α in α_, β in β_]

# Transfer Data to Workers
sendto(workers(), α=α, β=β, δ=δ,
    param_set=param_set,
    price_options=price_options,
    competition_params=competition_params
    )

function pmap_experiments()
    status = pmap(1:length(param_set)) do i
        try
            AlgorithmicCompetition.runCalvano(param_set[i][1], param_set[i][2], δ, price_options, competition_params)
        catch e
            @warn "failed to process $(i)"
            false # failure
        end
    end
end

@time pmap_experiments()


# TODOs
# - Add hook to save profit results
# - Get multiprocessing loop running
# - Run and save results


# init_matrix = hcat(fill([Q_0(price_options, δ)...], n_state_space)...)
# TODO: Parametrize init matrix, use above matrix as default

