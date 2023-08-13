# Patch to improve type stability and try to speed things up (avoid generator)
function RLBase.plan!(multiagent::MultiAgentPolicy, env::DDDCEnv)
    action_set = CartesianIndex{2}(
        RLBase.plan!(multiagent[Symbol(1)], env, Symbol(1)),
        RLBase.plan!(multiagent[Symbol(2)], env, Symbol(2)),
    )
    return action_set
end

function Experiment(env::DDDCEnv; stop_on_convergence = true)
    RLCore.Experiment(
        DDDCPolicy(env),
        env,
        AIAPCStop(env; stop_on_convergence = stop_on_convergence),
        DDDCHook(env),
    )
end

function Base.run(hyperparameters::DDDCHyperParameters; stop_on_convergence = true)
    env = DDDCEnv(hyperparameters)
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

"""
    run_and_extract(hyperparameters::DDDCHyperParameters; stop_on_convergence = true)

Runs the experiment and returns the economic summary.
"""
function run_and_extract(hyperparameters::DDDCHyperParameters; stop_on_convergence = true)
    economic_summary(run(hyperparameters; stop_on_convergence = stop_on_convergence))
end
