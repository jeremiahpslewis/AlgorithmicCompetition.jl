@testset "Policy operation test" begin
    α = Float64(0.125)
    β = Float64(1e-5)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = 1000
    price_index = 1:n_prices

    competition_params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))

    competition_solution = CompetitionSolution(competition_params)

    hyperparameters = AIAPCHyperParameters(
        α,
        β,
        δ,
        max_iter,
        competition_solution;
        convergence_threshold = 1,
    )
    env = AIAPCEnv(hyperparameters)
    policy = AIAPCPolicy(env)

    # Test full policy exploration of states
    push!(policy, PreActStage(), env)
    n_ = Int(1e5)
    policy_runs = [[plan!(policy, env)...] for i = 1:n_]
    checksum_ = [sum(unique(policy_runs[j][i] for j = 1:n_)) for i = 1:2]
    @test all(checksum_ .== sum(1:env.n_prices))
end

@testset "policy push! and optimise! test" begin
    competition_params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))
    competition_solution = CompetitionSolution(competition_params)

    env =
        AIAPCHyperParameters(
            Float64(0.1),
            Float64(1e-4),
            0.95,
            Int(1e7),
            competition_solution,
        ) |> AIAPCEnv

    policy = AIAPCPolicy(env)

    @test maximum(policy[Symbol(1)].policy.learner.approximator.table) ≈ 6.278004857861001
    @test minimum(policy[Symbol(1)].policy.learner.approximator.table) ≈ 4.111178690372623

    approx_table = copy(policy.agents[Symbol(1)].policy.learner.approximator.table)
    # First three rounds

    # t=1
    push!(policy[Symbol(1)].trajectory, policy[Symbol(1)].cache, state(env, Symbol(1)))
    push!(policy, PreEpisodeStage(), env)
    push!(policy, PreActStage(), env)
    @test policy.agents[Symbol(1)].cache.reward == nothing
    @test policy.agents[Symbol(1)].cache.terminal == nothing
    @test length(policy.agents[Symbol(1)].trajectory.container) == 0
    optimise!(policy, PreActStage())
    approx_table_t_1 = copy(policy.agents[Symbol(1)].policy.learner.approximator.table)
    @test approx_table_t_1 == approx_table # test that optimise! in t=1 is a noop
    action = RLBase.plan!(policy, env)
    act!(env, action)
    @test length(policy.agents[Symbol(1)].trajectory.container) == 0 # test that trajectory has not been filled
    push!(policy, PostActStage(), env)
    @test length(policy.agents[Symbol(1)].trajectory.container) == 1
    optimise!(policy, PostActStage())
    reward_1 = copy(policy.agents[Symbol(1)].cache.reward)
    @test reward_1 != 0
    push!(policy, PostEpisodeStage(), env)
    cache_1 = deepcopy(policy.agents[Symbol(1)].cache)

    # t=2
    push!(policy, PreEpisodeStage(), env)
    push!(policy, PreActStage(), env)
    @test length(policy.agents[Symbol(1)].trajectory.container) == 1
    optimise!(policy, PreActStage())
    approx_table_t_2 = copy(policy.agents[Symbol(1)].policy.learner.approximator.table)
    @test approx_table_t_2 != approx_table_t_1 # test that optimise! in t=2 is not a noop   
    action = RLBase.plan!(policy, env)
    act!(env, action)
    push!(policy, PostActStage(), env)
    optimise!(policy, PostActStage())
    reward_2 = copy(policy.agents[Symbol(1)].cache.reward)
    @test reward_2 != reward_1
    push!(policy, PostEpisodeStage(), env)
    cache_2 = deepcopy(policy.agents[Symbol(1)].cache)

    # t=3
    push!(policy, PreEpisodeStage(), env)
    push!(policy, PreActStage(), env)
    @test length(policy.agents[Symbol(1)].trajectory.container) == 1
    optimise!(policy, PreActStage())
    approx_table_t_3 = copy(policy.agents[Symbol(1)].policy.learner.approximator.table)
    @test approx_table_t_2 != approx_table_t_3 # test that optimise! in t=2 is not a noop
    action = RLBase.plan!(policy, env)
    act!(env, action)
    push!(policy, PostActStage(), env)
    optimise!(policy, PostActStage())
    reward_3 = copy(policy.agents[Symbol(1)].cache.reward)
    @test reward_3 != reward_2
    push!(policy, PostEpisodeStage(), env)
    cache_3 = deepcopy(policy.agents[Symbol(1)].cache)
    @test (cache_2.action != cache_3.action) || (cache_1.action != cache_2.action)
end
