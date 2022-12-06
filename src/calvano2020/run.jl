using ReinforcementLearning

function runCalvano(calvano_params::CalvanoParams)
    env = CalvanoEnv(calvano_params) 

    calvano_experiment = Experiment(
        policy=CalvanoPolicy(env),
        env=env |> SequentialEnv,
        stop_condition=CalvanoStop(env.params),
        hook=CalvanoHook(env),
    )
    return run(calvano_experiment)
end

# Look into DistributedReinforcementLearning.jl for running grid of experiments
# Add hook `(hook::YourHook)(::PostExperimentStage, agent, env)` to save profit results!

function runCalvano(α, β, δ, price_options)
    calvano_params = CalvanoParams(
        α=α,
        β=β,
        δ=δ,
        n_players=2,
        memory_length=2,
        price_options=price_options,
        max_iter=Int(1e9),
        convergence_threshold=Int(1e5),
    )
    return runCalvano(calvano_params)        
end
