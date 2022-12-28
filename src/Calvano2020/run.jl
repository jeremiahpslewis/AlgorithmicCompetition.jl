using ReinforcementLearning

function setupCalvanoExperiment(env::CalvanoEnv)
    Experiment(
        policy = CalvanoPolicy(env),
        env = SequentialEnv(env),
        stop_condition = CalvanoStop(env),
        hook = CalvanoHook(env),
    )
end

function setupCalvanoExperiment(
    α::Float64,
    β::Float64,
    δ::Float64,
    price_options::Vector{Float64},
    competition_params::CompetitionParameters,
    p_Bert_nash_equilibrium::Float64,
    p_monop_opt::Float64;
    max_iter::Int = Int(1e9),
    convergence_threshold::Int = Int(1e5),
    profit_function = p -> π_fun(p, competition_params),
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
        p_Bert_nash_equilibrium::Float64,
        p_monop_opt::Float64,
    )
    return setupCalvanoExperiment(env)
end

runCalvano(env::CalvanoEnv) = setupCalvanoExperiment(env) |> run(; describe = false)

# TODO: Look into DistributedReinforcementLearning.jl for running grid of experiments
# Add hook `(hook::YourHook)(::PostExperimentStage, agent, env)` to save profit results!

function runCalvano(
    α::Float64,
    β::Float64,
    δ::Float64,
    price_options::Vector{Float64},
    competition_params::CompetitionParameters,
    p_Bert_nash_equilibrium::Float64,
    p_monop_opt::Float64;
    max_iter::Int = Int(1e9),
    convergence_threshold::Int = Int(1e5),
    profit_function = p -> π_fun(p, competition_params),
)
    experiment = setupCalvanoExperiment(
        α,
        β,
        δ,
        price_options,
        competition_params,
        max_iter = max_iter,
        convergence_threshold = convergence_threshold,
        profit_function = profit_function,
        p_Bert_nash_equilibrium::Float64,
        p_monop_opt::Float64,
    )
    return run(experiment; describe = false)
end

function buildParameterSet(; n_increments=100)
    # TODO: Fix this parameterization based on Calvano pg. 12
    α_ = range(0.025, 0.25, n_increments)
    β_ = range(1.25e-8, 2e-5, n_increments)

    competition_params = CompetitionParameters(
        0.25,
        0,
        [2, 2],
        [1, 1],
    )
    ξ = 0.1
    δ = 0.95
    n_prices = 15

    model_monop, p_monop = solve_monopolist(competition_params)

    p_Bert_nash_equilibrium = solve_bertrand(competition_params)[2][1]
    p_monop_opt = solve_monopolist(competition_params)[2][1]

    # p_monop defined above
    p_range_pad = ξ * (p_monop_opt - p_Bert_nash_equilibrium)
    price_options = [range(p_Bert_nash_equilibrium, p_monop_opt, n_prices)...]

    return [
        (α, β, δ, price_options, competition_params, p_Bert_nash_equilibrium, p_monop_opt)
        for α in α_ for β in β_
    ]
end
