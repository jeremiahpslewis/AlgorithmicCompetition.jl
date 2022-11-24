using ReinforcementLearning
	
include("src/calvano2020/params.jl")
include("src/calvano2020/envs.jl")
include("src/calvano2020/hooks.jl")
include("src/calvano2020/stop.jl")
include("src/calvano2020/policy.jl")

calvano_params = CalvanoParams(
    α=α,
    β=β,
    δ=δ,
    n_players=2,
    memory_length=2, 
    price_options=price_options
    max_iter=1e9,
)


function runCalvano(calvano_params::CalvanoParams)
    init_matrix = fill(0.0, n_prices, n_state_space)
    calvano_init_matrix = zeros(calvano_params.n_prices, calvano_params.n_prices)
    calvano_env = CalvanoEnv(calvano_params)
    calvano_policy = CalvanoPolicy(env::CalvanoEnv, init_matrix::Matrix, params::CalvanoParams)
    calvano_hook = CalvanoHook(env, convergence_check)
    run(multi_agent_policy,
        env,
        
    )
end

# Use struct `Experiment` ...{Symbol(s)}(ex.policy, ex.env, ex.stop_condition, ex.hook)
# to collect experiment objects
# Look into DistributedReinforcementLearning.jl for running grid of experiments
# Add hook `(hook::YourHook)(::PostExperimentStage, agent, env)` to save profit results!


n_iter = Int(5e6)
convergence_threshold = Int(1e5)
# Scale up to 500,000, implement stop after convergence signal!
convergence_check = ConvergenceCheck(n_state_space=n_state_space, n_players=n_players)

@time 
