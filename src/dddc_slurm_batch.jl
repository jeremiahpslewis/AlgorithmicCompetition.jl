using AlgorithmicCompetition
using Dates

if Sys.isapple()
    # For debugging on MacOS
    ENV["DEBUG"] = 1
    ENV["SLURM_ARRAY_TASK_ID"] = 1
    ENV["SLURM_ARRAY_JOB_ID"] = 1
    ENV["N_ITERATIONS"] = 1
    ENV["VERSION"] = "v1"
end

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
        n_grid_increments = 1,
        batch_metadata = (
            SLURM_ARRAY_JOB_ID = SLURM_ARRAY_JOB_ID,
            SLURM_ARRAY_TASK_ID = SLURM_ARRAY_TASK_ID,
        ),
        debug = debug,
    )
end

