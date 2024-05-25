using AlgorithmicCompetition
using Dates


debug = parse(Int, ENV["DEBUG"]) == 1
SLURM_ARRAY_TASK_ID = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
SLURM_ARRAY_JOB_ID = parse(Int, ENV["SLURM_ARRAY_JOB_ID"])
n_parameter_iterations = parse(Int, ENV["N_ITERATIONS"])

if debug && SLURM_ARRAY_TASK_ID > 10
    return
else
    AlgorithmicCompetition.run_dddc(
        version = ENV["VERSION"],
        start_timestamp = now(),
        n_parameter_iterations = n_parameter_iterations,
        batch_metadata = (
            SLURM_ARRAY_JOB_ID = SLURM_ARRAY_JOB_ID,
            SLURM_ARRAY_TASK_ID = SLURM_ARRAY_TASK_ID,
        ),
        debug = debug,
    )
end

