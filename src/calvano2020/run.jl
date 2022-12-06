using ReinforcementLearning
	
include("calvano2020/params.jl")
include("calvano2020/envs.jl")
include("calvano2020/hooks.jl")
include("calvano2020/stop.jl")
include("calvano2020/policy.jl")

function runCalvano(calvano_params::CalvanoParams)
    
    env = CalvanoEnv(calvano_params)

    calvano_experiment = Experiment(
        policy=CalvanoPolicy(env),
        env=env,
        stop_condition=CalvanoStop(env.params),
        hook=CalvanoHook(env)
    )
    run(calvano_experiment)
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
        max_iter=1e9,
        convergence_threshold=1e5,
    )
    runCalvano(calvano_params)        
end
