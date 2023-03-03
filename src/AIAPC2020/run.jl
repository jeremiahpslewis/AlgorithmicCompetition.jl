using ReinforcementLearningCore
using Distributed

function Experiment(env::AIAPCEnv; stop_on_convergence = true)
    Experiment(
        policy = AIAPCPolicy(env),
        env = SequentialEnv(env),
        stop_condition = AIAPCStop(env; stop_on_convergence = stop_on_convergence),
        hook = AIAPCHook(env),
    )
end

function run(experiments::Vector{ReinforcementLearningCore.Experiment})
    sendto(workers(), experiments = experiments)
    status = pmap(1:length(experiments)) do i
        Base.run(experiments[i]; describe = false)
    end
end

function run(hyperparameters::AIAPCHyperParameters; stop_on_convergence = true)
    env = AIAPCEnv(hyperparameters)
    experiment = Experiment(env; stop_on_convergence = stop_on_convergence)
    return Base.run(experiment; describe = false)
end

function run_and_extract(hyperparameters::AIAPCHyperParameters; stop_on_convergence = true)
    economic_summary(run(hyperparameters; stop_on_convergence = stop_on_convergence))
end
