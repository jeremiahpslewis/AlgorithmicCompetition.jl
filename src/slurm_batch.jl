using AlgorithmicCompetition
using Dates

AlgorithmicCompetition.run_aiapc(
    version = ENV["VERSION"],
    start_timestamp = now(),
    n_parameter_iterations = parse(Int, ENV["N_ITERATIONS"]),
    parameter_index = parse(Int, ENV["SLURM_ARRAY_TASK_ID"]),
)
