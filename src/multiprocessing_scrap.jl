using Distributed
using AlgorithmicCompetition: AlgorithmicCompetition, CalvanoHyperParameters, CalvanoEnv, Experiment
using ReinforcementLearning
using Chain

multiproc = true

using ParallelDataTransfer

_procs = addprocs(1,
    topology=:master_worker,
    exeflags=["--threads=1", "--project=$(Base.active_project())"]
    )

@everywhere begin
    using Pkg; Pkg.instantiate()
    using AlgorithmicCompetition: AlgorithmicCompetition, CalvanoHyperParameters, CalvanoEnv, run
end

n_increments = 100
max_iter = Int(1e6) # Should be 1e9
α_ = range(0.025, 0.25, n_increments)
β_ = range(1.25e-8, 2e-5, n_increments) 
δ = 0.95

hyperparameter_vect = [CalvanoHyperParameters.(α,  β, δ, max_iter) for α in α_ for β in β_]

pmap(AlgorithmicCompetition.run, param_set)
