using ReinforcementLearning

function runCalvano(env::CalvanoEnv)
    calvano_experiment = Experiment(
        policy=CalvanoPolicy(env),
        env=SequentialEnv(env),
        stop_condition=CalvanoStop(env),
        hook=CalvanoHook(env),
    )
    return run(calvano_experiment)
end

# Look into DistributedReinforcementLearning.jl for running grid of experiments
# Add hook `(hook::YourHook)(::PostExperimentStage, agent, env)` to save profit results!

function runCalvano(
        α::Float64,
        β::Float64,
        δ::Float64,
        price_options::Base.AbstractVecOrTuple{Float64},
        competition_params::CompetitionParameters;
        max_iter::Int = Int(1e9),
        convergence_threshold::Int = Int(1e5),
    )
    env = CalvanoEnv(
        α=α,
        β=β,
        δ=δ,
        n_players=2,
        memory_length=2,
        price_options=price_options,
        max_iter=max_iter,
        convergence_threshold=convergence_threshold,
        profit_function=(p_1, p_2) -> π_fun(p_1, p_2, competition_params)
    )
    return runCalvano(env)
end
