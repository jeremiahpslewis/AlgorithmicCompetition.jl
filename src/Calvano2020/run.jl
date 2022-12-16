using ReinforcementLearning

function setupCalvanoExperiment(env::CalvanoEnv)
    Experiment(
        policy=CalvanoPolicy(env),
        env=SequentialEnv(env),
        stop_condition=CalvanoStop(env),
        hook=CalvanoHook(env),
    )
end

function setupCalvanoExperiment(
    α::Float64,
    β::Float64,
    δ::Float64,
    price_options::Vector{Float64},
    competition_params::CompetitionParameters;
    max_iter::Int=Int(1e9),
    convergence_threshold::Int=Int(1e5),
    profit_function=(p_1, p_2) -> π_fun([p_1, p_2], competition_params),
)

    env = CalvanoEnv(
        α,
        β,
        δ,
        2, # n_players
        2, # memory_length
        price_options,
        max_iter,
        convergence_threshold,
        profit_function,
    )
    return setupCalvanoExperiment(env)
end

runCalvano(env::CalvanoEnv) = setupCalvanoExperiment(env) |> run(; describe=false)

# Look into DistributedReinforcementLearning.jl for running grid of experiments
# Add hook `(hook::YourHook)(::PostExperimentStage, agent, env)` to save profit results!

function runCalvano(
    α::Float64,
    β::Float64,
    δ::Float64,
    price_options::Vector{Float64},
    competition_params::CompetitionParameters;
    max_iter::Int=Int(1e9),
    convergence_threshold::Int=Int(1e5),
    profit_function=(p_1, p_2) -> π_fun(SA[p_1, p_2], competition_params)
)
    experiment = setupCalvanoExperiment(
        α,
        β,
        δ,
        price_options,
        competition_params,
        max_iter=max_iter,
        convergence_threshold=convergence_threshold,
        profit_function=profit_function,
    )
    return run(experiment; describe=false)
end
