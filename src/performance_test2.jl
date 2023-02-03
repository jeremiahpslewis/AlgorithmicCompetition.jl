# using Distributed
# using ParallelDataTransfer
using AlgorithmicCompetition: buildParameterSet, setupCalvanoExperiment, run
using ReinforcementLearning

# n_proc = 2         
# addprocs(n_proc,
#     topology=:master_worker,
#     exeflags=["--threads=1", "--project=$(Base.active_project())"]
# )

# @everywhere begin
#     using Pkg; Pkg.instantiate()
#     using AlgorithmicCompetition
# end

params_ = buildParameterSet(n_increments=2)
experiments_ = [setupCalvanoExperiment(p...) for p in params_]
Base.run(experiments_[1])

# try
#     ReinforcementLearning.run(experiments_[1])
# finally
#     rmprocs(_procs)
# end

# runCalvanoSimulation()
