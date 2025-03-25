using UUIDs

@inline function RLBase.plan!(multiagent::MultiAgentPolicy, env::DDDCEnv)
    @inbounds return CartesianIndex{2}(
        RLBase.plan!(multiagent[Player(1)], env, Player(1)),
        RLBase.plan!(multiagent[Player(2)], env, Player(2)),
    )
end

@inline function Experiment(env::DDDCEnv; stop_on_convergence = true)
    RLCore.Experiment(
        DDDCPolicy(env),
        env,
        AIAPCStop(env; stop_on_convergence = stop_on_convergence),
        DDDCHook(env),
    )
end

@inline function Base.run(hyperparameters::DDDCHyperParameters; stop_on_convergence = true)
    env = DDDCEnv(hyperparameters)
    experiment = Experiment(env; stop_on_convergence = stop_on_convergence)
    RLCore._run(
        experiment.policy,
        experiment.env,
        experiment.stop_condition,
        experiment.hook,
        ResetIfEnvTerminated(),
    )
    return experiment
end

"""
    run_and_extract(hyperparameters::DDDCHyperParameters; stop_on_convergence = true)

Runs the experiment and returns the economic summary.
"""
function run_and_extract(
    hyperparameters::DDDCHyperParameters;
    stop_on_convergence = true,
    write_to_file_return_none = false,
    write_to_file_path = "data"
)
    @info "Running single simulation with hyperparameters: $hyperparameters"
    single_run_output = economic_summary(run(hyperparameters; stop_on_convergence = stop_on_convergence))

    if write_to_file_return_none
        single_run_df = extract_sim_results([single_run_output])

        Arrow.write(joinpath(write_to_file_path, string(UUIDs.uuid4()) * ".arrow"), single_run_df)
    
        return nothing
    else
        return single_run_output
    end
end
