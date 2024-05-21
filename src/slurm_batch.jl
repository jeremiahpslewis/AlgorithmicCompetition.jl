using AlgorithmicCompetition

AlgorithmicCompetition.run_aiapc(
    batch_size = batch_size,
    version = version,
    start_timestamp = start_timestamp,
    n_parameter_iterations = n_parameter_iterations,
    parameter_index = parse(Int, ENV["SLURM_ARRAY_TASK_ID"]),
)
