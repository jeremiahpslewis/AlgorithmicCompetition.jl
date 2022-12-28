using Distributed
using ParallelDataTransfer
using AlgorithmicCompetition
using ReinforcementLearning

function run(experiments::Vector{Experiment})
    sendto(workers(),
        experiments=experiments,
    )
    status = pmap(1:length(experiments)) do i
        try
            run(experiments[i]; describe=false)
        catch e
            @warn "failed to process $(i)"
            false # failure
        end
    end
end

@everywhere begin
    using Pkg; Pkg.instantiate()
    using AlgorithmicCompetition
end

try
    n_proc = 2
    _procs = addprocs(n_proc,
        topology=:master_worker,
        exeflags=["--threads=1", "--project=$(Base.active_project())"]
        )

    params_ = buildParameterSet(n_increments=2)
    experiments_ = setupCalvanoExperiment(params_...)
    run(experiments_)
finally
    rmprocs(_procs)
end
