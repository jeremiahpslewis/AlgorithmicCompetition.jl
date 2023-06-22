@testset "EpsilonGreedy" begin
    explorer = EpsilonGreedyExplorer(
        kind = :exp,
        ϵ_init = 1,
        ϵ_stable = 0,
        decay_steps = Int(round(1 / 1e-5)),
    )
    @test RLCore.get_ϵ(explorer, 1e5) ≈ 0.36787944117144233 # Percentage according to formula and paper convergence results
    @test_broken RLCore.get_ϵ(explorer, 1e5) ≈ 0.1353352832366127 # Percentage cited in AIAPC paper (2x step count)

    explorer = AIAPCEpsilonGreedyExplorer(Float64(1e-5))
    @test get_ϵ(explorer, 1e5) ≈ 0.36787944117144233 # Percentage according to formula and paper convergence results
    @test_broken get_ϵ(explorer, 1e5) ≈ 0.1353352832366127 # Percentage cited in AIAPC paper (2x step count)
end
