using ReinforcementLearning
using Distributed

function Experiment(env::AIAPCEnv; stop_on_convergence=true)
    ReinforcementLearning.Experiment(
        policy = AIAPCPolicy(env),
        env = SequentialEnv(env),
        stop_condition = AIAPCStop(env; stop_on_convergence=stop_on_convergence),
        hook = AIAPCHook(env),
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

function run(hyperparameter_vect::Vector{AIAPCHyperParameters}; stop_on_convergence=true)
    pmap(x -> run(x; stop_on_convergence=stop_on_convergence), hyperparameter_vect)
end

function run(hyperparameters::AIAPCHyperParameters; stop_on_convergence=true)
    env = AIAPCEnv(hyperparameters)
    experiment = Experiment(env; stop_on_convergence=stop_on_convergence)
    return Base.run(experiment)
end

function run_and_extract(hyperparameters::AIAPCHyperParameters; stop_on_convergence=true)
    economic_summary(run(hyperparameters; stop_on_convergence=stop_on_convergence))
end
