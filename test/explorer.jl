using ReinforcementLearningFarm: EpsilonSpeedyExplorer
using ReinforcementLearningFarm

@testset "EpsilonGreedy" begin
    # This yields a different result (same result, but at 2x step count) than in the paper for 100k steps, but the same convergece duration at α and β midpoints 850k (pg. 13)
    explorer = EpsilonSpeedyExplorer(Float64(1e-5))
    @test RLFarm.get_ϵ(explorer, Int(1e5)) ≈ 0.36787944117144233 # Percentage according to formula and paper convergence results
    @test_broken RLFarm.get_ϵ(explorer, Int(1e5)) ≈ 0.1353352832366127 # Percentage cited in AIAPC paper (2x step count)
end
