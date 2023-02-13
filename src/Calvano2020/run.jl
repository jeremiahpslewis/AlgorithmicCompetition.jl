using ReinforcementLearning
using Distributed
using ParallelDataTransfer

function Experiment(env::CalvanoEnv)
    ReinforcementLearning.Experiment(
        policy = CalvanoPolicy(env),
        env = SequentialEnv(env),
        stop_condition = CalvanoStop(env; stop_on_convergence=false), # TODO: reactivate stop on convergence
        hook = CalvanoHook(env),
    )
end

function run(experiments::Vector{ReinforcementLearningCore.Experiment})
    sendto(workers(), experiments = experiments)
    status = pmap(1:length(experiments)) do i
        # try
        Base.run(experiments[i])
        # catch e
        #     @warn "failed to process $(i)"
        #     false # failure
        # end
    end
end

function run(hyperparameter_vect::Vector{CalvanoHyperParameters})
    pmap(run, hyperparameters)
end

function run(hyperparameters::CalvanoHyperParameters)
    env = CalvanoEnv(hyperparameters)
    experiment = Experiment(env)
    return Base.run(experiment)
end

function run_and_extract(hyperparameters::CalvanoHyperParameters)
    economic_summary(run(hyperparameters))
end
