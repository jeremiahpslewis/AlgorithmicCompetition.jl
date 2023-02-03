using ReinforcementLearning
using Distributed
using ParallelDataTransfer

function Experiment(env::CalvanoEnv)
    ReinforcementLearning.Experiment(
        policy=CalvanoPolicy(env),
        env=SequentialEnv(env),
        stop_condition=CalvanoStop(env),
        hook=CalvanoHook(env),
    )
end

function run(experiments::Vector{ReinforcementLearningCore.Experiment})
    sendto(workers(),
        experiments=experiments,
    )
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
    CalvanoEnv() |> Experiment()
end
