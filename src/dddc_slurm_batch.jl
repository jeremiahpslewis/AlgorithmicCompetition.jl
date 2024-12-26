using AlgorithmicCompetition
using Dates
using Distributed

params = AlgorithmicCompetition.extract_params_from_environment()

# AlgorithmicCompetition.setup_logger(params)

@info "Parameters: $params"

@info "Running DDDC batch."

if params[:n_cores] > 1
    _procs = addprocs(
        params[:n_cores],
        topology = :master_worker,
        exeflags = ["--threads=1", "--project=$(Base.active_project())"],
    )

    @everywhere begin
        @info "Load dependencies on all processes."
        using Pkg
        Pkg.instantiate()
        using AlgorithmicCompetition:
            extract_params_from_environment, setup_logger, run_and_extract
        params = extract_params_from_environment()
        # setup_logger(params)
        @info "Dependencies loaded."
    end
end

if params[:debug] && params[:SLURM_ARRAY_TASK_ID] > 10
    return
else
    @info "Running DDDC batch with n_grid_increments = $(params[:n_grid_increments])."
    AlgorithmicCompetition.run_dddc(
        version = ENV["VERSION"],
        start_timestamp = now(),
        n_parameter_iterations = params[:n_parameter_iterations],
        n_grid_increments = params[:n_grid_increments],
        batch_size = 1,
        batch_metadata = (
            SLURM_ARRAY_JOB_ID = params[:SLURM_ARRAY_JOB_ID],
            SLURM_ARRAY_TASK_ID = params[:SLURM_ARRAY_TASK_ID],
        ),
        max_iter = params[:max_iter],
        convergence_threshold = params[:convergence_threshold],
        debug = params[:debug],
    )
end
