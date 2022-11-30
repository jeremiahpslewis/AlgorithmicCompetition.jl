using Test


@testset "Competitive Equilibrium: Monopoly" begin
    params = CompetitionParameters(μ = 0.25, a_0 = 0, a = [a_0, 2, 2], c = [1, 1], n_firms = 2)
	model_monop, p_monop = solve_monopolist(params)

    # symmetric solution found
    @test value(p_monop[1]) ≈ value(p_monop[2])

    # Match Calvano 2020 parameterization
	@test isapprox(value(p_monop[1]), 1.92498; atol=0.0001)
	p_monop_opt = value(p_monop[2])
end



@testset "Competitive Equilibrium: Bertrand" begin
    p_Bertrand_ = value.(solve_bertrand(params)[2])
    p_Bertrand = p_Bertrand_[1]

    # Parameter recovery
    @assert isapprox(p_Bertrand_[2], 1.47293; atol=1e-3)

    # Best response function matches Calvano 2020
    @assert isapprox(p_BR(1.47293), 1.47293; atol=1e6)
end
