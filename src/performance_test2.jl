using Distributed
using ParallelDataTransfer
using AlgorithmicCompetition: buildParameterSet, setupCalvanoExperiment, run
using ReinforcementLearning

function runCalvanoSimulation()
    # n_proc = 2
    # _procs = addprocs(n_proc,
    #     topology=:master_worker,
    #     exeflags=["--threads=1", "--project=$(Base.active_project())"]
    # )
    # include("everywhere_snippet.jl")

    params_ = buildParameterSet(n_increments=2)
    experiments_ = [setupCalvanoExperiment(p...) for p in params_]

    try
        ReinforcementLearning.run(experiments_[1])
    finally
        # rmprocs(_procs)
    end
end

runCalvanoSimulation()
