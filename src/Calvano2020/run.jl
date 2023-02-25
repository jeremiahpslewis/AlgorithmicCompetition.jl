using ReinforcementLearning
using Distributed

function Experiment(env::CalvanoEnv; stop_on_convergence=true)
    ReinforcementLearning.Experiment(
        policy = CalvanoPolicy(env),
        env = SequentialEnv(env),
        stop_condition = CalvanoStop(env; stop_on_convergence=stop_on_convergence),
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

function run(hyperparameter_vect::Vector{CalvanoHyperParameters}; stop_on_convergence=true)
    pmap(x -> run(x; stop_on_convergence=stop_on_convergence), hyperparameter_vect)
end

function run(hyperparameters::CalvanoHyperParameters; stop_on_convergence=true)
    env = CalvanoEnv(hyperparameters)
    experiment = Experiment(env; stop_on_convergence=stop_on_convergence)
    return Base.run(experiment)
end

function run_and_extract(hyperparameters::CalvanoHyperParameters; stop_on_convergence=true)
    economic_summary(run(hyperparameters; stop_on_convergence=stop_on_convergence))
end
