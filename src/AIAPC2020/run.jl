using ReinforcementLearningCore
using ReinforcementLearningEnvironments
using Distributed
import Base
import ReinforcementLearningBase: RLBase
import ReinforcementLearningCore: RLCore

# Patch to improve type stability and try to speed things up (avoid generator)
function RLBase.plan!(multiagent::MultiAgentPolicy, env::AIAPCEnv)
    return CartesianIndex{2}(
        Tuple{Int64,Int64}(RLBase.plan!(multiagent[player_], env, player_) for player_ ∈ players(env))
    )
end

function Experiment(env::AIAPCEnv; stop_on_convergence = true, debug = false)
    RLCore.Experiment(
        AIAPCPolicy(env),
        env,
        AIAPCStop(env; stop_on_convergence = stop_on_convergence),
        debug ? AIAPCDebugHook(env) : AIAPCPerformanceHook(env),
    )
end

function Base.run(experiments::Vector{ReinforcementLearningCore.Experiment})
    sendto(workers(), experiments = experiments)
    status = pmap(1:length(experiments)) do i
        experiment = experiment[i]
        RLCore._run(
            experiment.policy,
            experiment.env,
            experiment.stop_condition,
            experiment.hook,
            ResetAtTerminal(),
        )
        experiments[i]
    end
end

function Base.run(hyperparameters::AIAPCHyperParameters; stop_on_convergence = true, debug = false)
    env = AIAPCEnv(hyperparameters)
    experiment = Experiment(env; stop_on_convergence = stop_on_convergence, debug = debug)
    RLCore._run(
        experiment.policy,
        experiment.env,
        experiment.stop_condition,
        experiment.hook,
        ResetAtTerminal(),
    )
    return experiment
end

"""
    run_and_extract(hyperparameters::AIAPCHyperParameters; stop_on_convergence = true)

Runs the experiment and returns the economic summary.
"""
function run_and_extract(hyperparameters::AIAPCHyperParameters; stop_on_convergence = true, debug = false)
    economic_summary(run(hyperparameters; stop_on_convergence = stop_on_convergence, debug = debug))
end
