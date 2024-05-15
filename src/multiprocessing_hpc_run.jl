using Distributed
using ClusterManagers
using AlgorithmicCompetition
using Dates
using CSV

n_parameter_iterations = 2
n_parameter_combinations = 100 #00
batch_size = 500
duration = 0.3 # in hours, e.g. 8 hours per run
duration_minutes = Int(floor(duration * 60))
n_sims_per_hour = 12 * 60 # 12 simulations per minute
speed_discount = 0.9 # 10% buffer for speed discount
n_cores = n_parameter_iterations * n_parameter_combinations / duration / n_sims_per_hour * speed_discount |> ceil |> Int

version = "v0.0.2"
start_timestamp = now()
start_timestamp_str = Dates.format(start_timestamp, "yyyy-mm-dd_HHMMSS")

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
    n_parameter_iterations=n_parameter_iterations,
    α_range=Float64.(range(0.0025, 0.25, 5)),
    β_range=Float64.(range(0.02, 2, 5)),    
)


file_name = "$(ENV["HOME"])/simulation_results_aiapc_$(version)_$(start_timestamp_str).csv"
CSV.write(file_name, exp_df)

rmprocs(workers())


