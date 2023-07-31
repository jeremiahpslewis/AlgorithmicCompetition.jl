using AlgorithmicCompetition: TotalRewardPerEpisodeLastN
using ReinforcementLearningEnvironments
using ReinforcementLearningBase
using ReinforcementLearningCore

@testset "TotalRewardPerEpisodeLastN" begin
    @testset "Single Agent" begin
        hook = TotalRewardPerEpisodeLastN(max_steps = 10)
        env = TicTacToeEnv()
        agent = RandomPolicy()

        for i = 1:15
            push!(hook, PostActStage(), agent, env)
            push!(hook, PostEpisodeStage(), agent, env)
            @test length(hook.rewards) == min(i, 10)
            @test hook.rewards[min(i, 10)] == reward(env)
        end
    end

    @testset "MultiAgent" begin
        hook = TotalRewardPerEpisodeLastN(max_steps = 10)
        env = TicTacToeEnv()
        agent = RandomPolicy()

        for i = 1:15
            push!(hook, PostActStage(), agent, env, :Cross)
            push!(hook, PostEpisodeStage(), agent, env, :Cross)
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
    exper = Experiment(env)
    state(env, Symbol(1))
    policies = AIAPCPolicy(env, mode = "zero")
    push!(exper.hook[Symbol(1)][2], Int16(2), Int8(3), false)
    @test exper.hook[Symbol(1)][2].best_response_vector[2] == 3

    policies[Symbol(1)].policy.learner.approximator.table[11, :] .= 10
    push!(
        exper.hook[Symbol(1)][2],
        PostActStage(),
        policies[Symbol(1)],
        exper.env,
        Symbol(1),
    )
    @test exper.hook[Symbol(1)][2].best_response_vector[state(env, Symbol(1))] == 11
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
    push!(convergence_hook, PostActStage(), policies[Symbol(1)], env, Symbol(1))
    @test convergence_hook.convergence_duration == 0
    @test convergence_hook.iterations_until_convergence == 1
    @test convergence_hook.best_response_vector[state(env, Symbol(1))] != 0
    @test convergence_hook.is_converged != true

    convergence_hook_1 = ConvergenceCheck(env.n_state_space, 1)
    convergence_hook_1.best_response_vector = Vector{Int}(fill(8, 225))
    push!(convergence_hook_1, PostActStage(), policies[Symbol(1)], env, Symbol(1))

    @test convergence_hook.iterations_until_convergence == 1
    @test convergence_hook.convergence_duration ∈ [0, 1]
    # @test convergence_hook_1.is_converged == true
end
