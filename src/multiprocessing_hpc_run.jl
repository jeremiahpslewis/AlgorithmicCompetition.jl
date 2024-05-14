using Distributed, ClusterManagers, AlgorithmicCompetition, Dates

n_cores = 1000
n_iter = 100
batch_size = Int(floor(1e4*n_iter / n_cores))
duration = Int(ceil(batch_size / 6)) # in minutes, appproximately 6 simulations per minute are possible
version = "v0.0.2"
start_timestamp = now()
start_timestamp = Dates.format(start_timestamp, "yyyy-mm-dd HHMMSS")

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
    start_timestamp=start_timestamp
)


file_name = "simulation_results_aiapc_$(version)_$(start_timestamp).csv"
CSV.write(file_name, exp_df)

rmprocs(workers())


