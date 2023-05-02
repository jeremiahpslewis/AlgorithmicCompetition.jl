using ReinforcementLearningCore
using ReinforcementLearningEnvironments
using Distributed
import Base

function Experiment(env::AIAPCEnv; stop_on_convergence = true)
    RLCore.Experiment(
        AIAPCPolicy(env),
        env,
        AIAPCStop(env; stop_on_convergence = stop_on_convergence),
        AIAPCHook(env),
    )
end

function Base.run(experiments::Vector{ReinforcementLearningCore.Experiment})
    sendto(workers(), experiments = experiments)
    status = pmap(1:length(experiments)) do i
        Base.run(experiments[i])
        experiments[i]
    end
end

function Base.run(hyperparameters::AIAPCHyperParameters; stop_on_convergence = true)
    env = AIAPCEnv(hyperparameters)
    experiment = Experiment(env; stop_on_convergence = stop_on_convergence)
    return run(experiment)
end

function run_and_extract(hyperparameters::AIAPCHyperParameters; stop_on_convergence = true)
    economic_summary(run(hyperparameters; stop_on_convergence = stop_on_convergence))
end
