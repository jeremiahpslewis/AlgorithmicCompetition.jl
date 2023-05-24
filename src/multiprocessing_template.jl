using AlgorithmicCompetition
using Chain
using Statistics
using DataFrames
using DataFrameMacros
using CSV
using Distributed
using CSV

n_procs_ = 40 # up to 8 performance cores on m1

_procs = addprocs(
    n_procs_,
    topology = :master_worker,
    exeflags = ["--threads=1", "--project=$(Base.active_project())"],
)

@everywhere begin
    using Pkg
    Pkg.instantiate()
    using AlgorithmicCompetition: run_and_extract
end

@time exp_list = AlgorithmicCompetition.run_aiapc(;
    n_parameter_iterations = 10,
    n_parameter_increments = 10,
)

rmprocs(_procs)

α_result = [ex.α for ex in exp_list if !(ex isa Exception)]
β_result = [ex.β for ex in exp_list if !(ex isa Exception)]
iterations_until_convergence =
    [ex.iterations_until_convergence for ex in exp_list if !(ex isa Exception)]

avg_profit_result = [ex.avg_profit[1] for ex in exp_list if !(ex isa Exception)]

df = DataFrame(
    α = α_result,
    β = β_result,
    π_bar = avg_profit_result,
    iterations_until_convergence = iterations_until_convergence,
)

CSV.write("simulation_results.csv", df)
