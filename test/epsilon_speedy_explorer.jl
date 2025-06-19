# Source: https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/701d2f976bc40d10f0b5b751e237f3ce01b228ff/src/ReinforcementLearningFarm/test/algorithms/explorers/epsilon_speedy_explorer.jl

@testset "EpsilonSpeedyExplorer2" begin
    using Test

    @testset "EpsilonSpeedyExplorer2" begin
        @testset "constructor" begin
            explorer = EpsilonSpeedyExplorer2(0.1)
            @test explorer.β == 0.1
            @test explorer.β_neg == -0.1
            @test explorer.step[] == 1
            @test explorer.rng === Random.GLOBAL_RNG
        end
    
        @testset "get_ϵ" begin
            explorer = EpsilonSpeedyExplorer2(0.1)
            @test get_ϵ(explorer) ≈ exp(-0.1)
            explorer.step[] = 10
            @test get_ϵ(explorer) ≈ exp(-1.0)
        end
    
        @testset "plan" begin
            explorer = EpsilonSpeedyExplorer2(0.1)
            values = [1, 2, 3, 4, 5]
            mask = [true, false, true, false, true]
    
            @testset "without mask" begin
                action = RLBase.plan!(explorer, values)
                @test action ∈ 1:length(values)
            end
    
            @testset "with mask" begin
                action = RLBase.plan!(explorer, values, mask)
                @test action ∈ findall(mask)
            end
    
            @testset "with true mask" begin
                true_mask = [true, true, true, true, true]
                action = RLBase.plan!(explorer, values, true_mask)
                @test action ∈ findall(true_mask)
            end
        end
    
        @testset "prob" begin
            explorer = EpsilonSpeedyExplorer2(0.1)
            values = [1, 2, 3, 4, 5]
            mask = [true, false, true, false, true]
    
            @testset "without mask" begin
                prob_dist = RLBase.prob(explorer, values)
                @test prob_dist.p ≈  [0.1809674836071919, 0.1809674836071919, 0.1809674836071919, 0.1809674836071919, 0.2761300655712324]
            end
    
            @testset "with mask" begin
                prob_dist = RLBase.prob(explorer, values, mask)
                @test prob_dist.p ≈ [0.30161247267865315, 0.0, 0.30161247267865315, 0.0, 0.39677505464269364]
            end
    
            @testset "with true mask" begin
                true_mask = [true, true, true, true, true]
                prob_dist = RLBase.prob(explorer, values, true_mask)
                @test prob_dist.p ≈ [0.1809674836071919, 0.1809674836071919, 0.1809674836071919, 0.1809674836071919, 0.2761300655712324]
            end
        end
    end
    
    @testset "EpsilonSpeedyExplorer2 correctness" begin
        explorer = RLFarm.EpsilonSpeedyExplorer2(1e-5)
        explorer.step[] = Int(1e5)
        @test RLFarm.get_ϵ(explorer) ≈ 0.36787944117144233
    end

    @testset "EpsilonSpeedyExplorer2 with nonzero min_ϵ" begin
        explorer2 = EpsilonSpeedyExplorer22(0.1, min_ϵ=0.01)
        @test explorer2.β == 0.1
        @test explorer2.β_neg == -0.1
        @test explorer2.min_ϵ == 0.01
        @test explorer2.step[] == 1
        @test explorer2.rng === Random.GLOBAL_RNG
        explorer.step[] = Int(1e100)
        @test RLFarm.get_ϵ(explorer2) == 0.01
    end
end
