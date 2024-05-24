using AlgorithmicCompetition
using Dates

AlgorithmicCompetition.run_aiapc(
    version = ENV["VERSION"],
    start_timestamp = now(),
    n_parameter_iterations = parse(Int, ENV["N_ITERATIONS"]),
    slurm_metadata = (
        SLURM_ARRAY_JOB_ID = parse(Int, ENV["SLURM_ARRAY_JOB_ID"]),
        SLURM_ARRAY_TASK_ID = parse(Int, ENV["SLURM_ARRAY_TASK_ID"]),
    ),
    debug = (parse(Int, ENV["DEBUG"]) == 1),
)
