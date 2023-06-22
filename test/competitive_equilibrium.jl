@testset "Competitive Equilibrium: Monopoly" begin
    params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))
    model_monop, p_monop = solve_monopolist(params)

    # symmetric solution found
    @test value(p_monop[1]) ≈ value(p_monop[2])

    # Match AIAPC 2020 parameterization
    @test value(p_monop[1]) ≈ 1.92498 atol = 0.0001
    p_monop_opt = value(p_monop[2])
end

@testset "Competitive Equilibrium: Bertrand" begin
    params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))

    p_Bertrand_ = value.(solve_bertrand(params)[2])
    p_Bertrand = p_Bertrand_[1]

    # Parameter recovery
    @test p_Bertrand_[2] ≈ 1.47293 atol = 1e-3

    # Best response function matches AIAPC 2020
    @test p_BR(1.47293, params) ≈ 1.47293 atol = 1e6
end

@testset "CompetitionParameters" begin
    @test CompetitionParameters(1, 1, (1.0, 1.0), (1.0, 1.0)) isa CompetitionParameters
    @test_throws DimensionMismatch CompetitionParameters(1, 1, (1.0, 1), (1.0))
end
