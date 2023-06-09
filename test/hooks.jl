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
            @test hook.rewards[i] == reward(env)
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
            @test hook.rewards[i] == reward(env, :Cross)
        end
    end

    @test hook.rewards[1] == hook[1]
end