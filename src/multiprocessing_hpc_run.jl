using Distributed
using ClusterManagers
using AlgorithmicCompetition
using Dates
using CSV

n_parameter_iterations = 10
n_parameter_combinations = 1000
batch_size = 500
duration = 1.5 # in hours, e.g. 8 hours per run
duration_minutes = Int(floor(duration * 60))
n_sims_per_hour = 5 * 60 # 5 simulations per minute
speed_discount = 0.9 # 10% buffer for speed discount
n_cores = n_parameter_iterations * n_parameter_combinations / duration / n_sims_per_hour * speed_discount |> ceil |> Int

version = "v0.0.2"
start_timestamp = now()
start_timestamp_str = Dates.format(start_timestamp, "yyyy-mm-dd HHMMSS")

addprocs(SlurmManager(n_cores), partition="normal", t="00:$duration_minutes:00", cpus_per_task="1", mem_per_cpu="1G")
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


file_name = "$(ENV["HOME"])/simulation_results_aiapc_$(version)_$(start_timestamp_str).csv"
CSV.write(file_name, exp_df)

rmprocs(workers())


