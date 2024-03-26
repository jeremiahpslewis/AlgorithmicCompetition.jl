using ReinforcementLearningFarm: TotalRewardPerLastNEpisodes
using ReinforcementLearning

@testset "TotalRewardPerLastNEpisodes" begin
    @testset "Single Agent" begin
        hook = TotalRewardPerLastNEpisodes(max_episodes = 10)
        env = TicTacToeEnv()
        agent = RandomPolicy()

        for i = 1:15
            push!(hook, PreEpisodeStage(), agent, env)
            push!(hook, PostActStage(), agent, env)
            @test length(hook.rewards) == min(i, 10)
            @test hook.rewards[min(i, 10)] == reward(env)
        end
    end

    @testset "MultiAgent" begin
        hook = TotalRewardPerLastNEpisodes(max_episodes = 10)
        env = TicTacToeEnv()
        agent = RandomPolicy()

        for i = 1:15
            push!(hook, PreEpisodeStage(), agent, env, :Cross)
            push!(hook, PostActStage(), agent, env, :Cross)
            @test length(hook.rewards) == min(i, 10)
            @test hook.rewards[min(i, 10)] == reward(env, :Cross)
        end
    end
end

@testset "Convergence Check Hook" begin
    competition_params_dict = Dict(
        :high => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
    )
    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

    env =
        AIAPCHyperParameters(
            Float64(0.1),
            Float64(1e-4),
            0.95,
            Int(1e7),
            competition_solution_dict,
        ) |> AIAPCEnv
    exper = Experiment(env; debug = true)
    state(env, Player(1))
    policies = AIAPCPolicy(env, mode = "zero")
    push!(exper.hook[Player(1)][1], Int64(2), Int64(3), false)
    @test exper.hook[Player(1)][1].best_response_vector[2] == 3

    policies[Player(1)].policy.learner.approximator.model[11, :] .= 10
    push!(
        exper.hook[Player(1)][1],
        PostActStage(),
        policies[Player(1)],
        exper.env,
        Player(1),
    )
    @test exper.hook[Player(1)][1].best_response_vector[state(env, Player(1))] == 11
end

@testset "ConvergenceCheck" begin
    competition_params_dict = Dict(
        :high => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
    )
    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

    env =
        AIAPCHyperParameters(
            Float64(0.1),
            Float64(2e-5),
            0.95,
            Int(1e7),
            competition_solution_dict,
        ) |> AIAPCEnv
    policies = env |> AIAPCPolicy

    convergence_hook = ConvergenceCheck(env.n_state_space, 1)
    push!(convergence_hook, PostActStage(), policies[Player(1)], env, Player(1))
    @test convergence_hook.convergence_duration == 0
    @test convergence_hook.iterations_until_convergence == 1
    @test convergence_hook.best_response_vector[state(env, Player(1))] != 0
    @test convergence_hook.is_converged != true

    convergence_hook_1 = ConvergenceCheck(env.n_state_space, 1)
    convergence_hook_1.best_response_vector = Vector{Int}(fill(8, 225))
    push!(convergence_hook_1, PostActStage(), policies[Player(1)], env, Player(1))

    @test convergence_hook.iterations_until_convergence == 1
    @test convergence_hook.convergence_duration ∈ [0, 1]
    # @test convergence_hook_1.is_converged == true
end
