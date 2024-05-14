using Distributed
using ClusterManagers
using AlgorithmicCompetition
using Dates
using CSV

n_parameter_iterations = 100
batch_size = 500
duration = # in minutes, appproximately 6 simulations per minute are possible
n_cores = Int(ceil(batch_size / 3)) 

Int(floor(1e4*n_parameter_iterations / n_cores))


version = "v0.0.2"
start_timestamp = now()
start_timestamp_str = Dates.format(start_timestamp, "yyyy-mm-dd HHMMSS")

addprocs(SlurmManager(n_cores), partition="normal", t="00:$duration:00", cpus_per_task="1", mem_per_cpu="1G")
# q="express"

@everywhere begin
    using Pkg
    Pkg.instantiate()
    using AlgorithmicCompetition: run_and_extract
end

@time exp_df = AlgorithmicCompetition.run_aiapc(
    batch_size=batch_size,
    version=version,
    start_timestamp=start_timestamp,
    max_iter=Int(1e3),
    convergence_threshold=Int(1e2),
    n_parameter_iterations=n_parameter_iterations,
)


file_name = "simulation_results_aiapc_$(version)_$(start_timestamp_str).csv"
CSV.write(file_name, exp_df)

rmprocs(workers())


