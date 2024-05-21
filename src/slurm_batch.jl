using AlgorithmicCompetition
using Dates

AlgorithmicCompetition.run_aiapc(
    version = "2024-05-21",
    start_timestamp = now(),
    n_parameter_iterations = 10,
    parameter_index = parse(Int, ENV["SLURM_ARRAY_TASK_ID"]),
)
