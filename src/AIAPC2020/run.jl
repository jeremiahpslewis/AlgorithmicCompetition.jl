using ReinforcementLearningCore
using ReinforcementLearningEnvironments
using Distributed
import Base

# Patch to improve type stability and try to speed things up (avoid generator)
function (multiagent::MultiAgentPolicy)(env::AIAPCEnv)
    return tuple(
        multiagent[Symbol(:1)](env, Symbol(:1)),
        multiagent[Symbol(:2)](env, Symbol(:1)),
    )
end

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
        experiment = experiment[i]
        RLCore._run(experiment.policy,
        experiment.env,
        experiment.stop_condition,
        experiment.hook,
        ResetAtTerminal())
        experiments[i]
    end
end

function Base.run(hyperparameters::AIAPCHyperParameters; stop_on_convergence = true)
    env = AIAPCEnv(hyperparameters)
    experiment = Experiment(env; stop_on_convergence = stop_on_convergence)
    RLCore._run(
        experiment.policy,
        experiment.env,
        experiment.stop_condition,
        experiment.hook,
        ResetAtTerminal(),
    )
    return experiment
end

function run_and_extract(hyperparameters::AIAPCHyperParameters; stop_on_convergence = true)
    economic_summary(run(hyperparameters; stop_on_convergence = stop_on_convergence))
end
