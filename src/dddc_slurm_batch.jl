using AlgorithmicCompetition
using Dates
using Distributed

if Sys.isapple()
    # For debugging on MacOS
    ENV["DEBUG"] = 1
    ENV["SLURM_ARRAY_TASK_ID"] = 1
    ENV["SLURM_ARRAY_JOB_ID"] = 1
    ENV["SLURM_CPUS_PER_TASK"] = 6
    ENV["VERSION"] = "v1"
end


debug = parse(Int, ENV["DEBUG"]) == 1
SLURM_ARRAY_TASK_ID = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
SLURM_ARRAY_JOB_ID = parse(Int, ENV["SLURM_ARRAY_JOB_ID"])
n_cores = parse(Int, ENV["SLURM_CPUS_PER_TASK"])
n_grid_increments = parse(Int, ENV["N_GRID_INCREMENTS"])

# Overrride in case of debugging
if debug && Sys.isapple()
    n_grid_increments = 1
elseif debug
    n_grid_increments = 10
end

if n_cores > 1
    _procs = addprocs(
        n_cores,
        topology = :master_worker,
        exeflags = ["--threads=1", "--project=$(Base.active_project())"],
    )

    @everywhere begin
        using Pkg
        Pkg.instantiate()
        using AlgorithmicCompetition: run_and_extract
    end
end

if debug && SLURM_ARRAY_TASK_ID > 10
    return
else
    AlgorithmicCompetition.run_dddc(
        version = ENV["VERSION"],
        start_timestamp = now(),
        n_parameter_iterations = 1,
        n_grid_increments = n_grid_increments,
        batch_size = 20,
        batch_metadata = (
            SLURM_ARRAY_JOB_ID = SLURM_ARRAY_JOB_ID,
            SLURM_ARRAY_TASK_ID = SLURM_ARRAY_TASK_ID,
        ),
        debug = debug,
    )
end
