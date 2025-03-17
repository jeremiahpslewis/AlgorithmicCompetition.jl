using Dates
using AlgorithmicCompetition
using Dates
using Distributed

ENV["N_PARAMETER_ITERATIONS"] = 100
ENV["N_GRID_INCREMENTS"] = 0
ENV["DEBUG"] = 0
ENV["VERSION"] = "v0.1.2"

params = AlgorithmicCompetition.extract_params_from_environment()

AlgorithmicCompetition.setup_logger(params)

@info "Parameters: $params"

@info "Running DDDC batch."

version = 0.1
start_timestamp = now()
start_timestamp = Dates.format(start_timestamp, "yyyy-mm-dd__HH_MM_SS")

n_procs_ = 7 # up to 8 performance cores on m1 (7 workers + 1 main)


_procs = addprocs(
    n_procs_,
    topology = :master_worker,
    exeflags = ["--threads=1", "--project=$(Base.active_project())"],
)

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
        setup_logger(params)
        @info "Dependencies loaded."
    end
end

@info "Running DDDC batch with n_grid_increments = $(params[:n_grid_increments])."
exp_list = AlgorithmicCompetition.run_dddc(
    version = params[:version],
    start_timestamp = now(),
    n_parameter_iterations = params[:n_parameter_iterations],
    n_grid_increments = params[:n_grid_increments],
    trembling_hand_parameters = [0.0, 0.001, 0.01, 0.1, 0.5],
    debug = params[:debug],
)

if params[:n_cores] > 1
    rmprocs(_procs)
end
