using Test


@testset "Competitive Equilibrium: Monopoly" begin
    params =
        CompetitionParameters(μ = 0.25, a_0 = 0, a = [a_0, 2, 2], c = [1, 1], n_firms = 2)
    model_monop, p_monop = solve_monopolist(params)

    # symmetric solution found
    @test value(p_monop[1]) ≈ value(p_monop[2])

    # Match Calvano 2020 parameterization
    @test isapprox(value(p_monop[1]), 1.92498; atol = 0.0001)
    p_monop_opt = value(p_monop[2])
end



@testset "Competitive Equilibrium: Bertrand" begin
    p_Bertrand_ = value.(solve_bertrand(params)[2])
    p_Bertrand = p_Bertrand_[1]

    # Parameter recovery
    @assert isapprox(p_Bertrand_[2], 1.47293; atol = 1e-3)

    # Best response function matches Calvano 2020
    @assert isapprox(p_BR(1.47293), 1.47293; atol = 1e6)
end

@testset "map_memory_to_state" begin
    n_prices = 15
    n_players = 2
    memory_length = 1
    n_state_space = n_prices ^ (n_players * memory_length)
    @test map_memory_to_state(repeat([n_prices], n_players), n_prices) == n_state_space
    @test map_memory_to_state(Array{Int,2}(repeat([n_prices], n_players)'), n_prices) == n_state_space
end

@testset "runCalvano full simulation" begin
    p_monop_opt = 1.92498
    p_Bert_nash_equilibrium = 1.47

    α = 0.125
    β = 1e-5
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    price_index = 1:n_prices

    competition_params = CompetitionParameters(
        μ = 0.25,
        a_0 = 0,
        a = [2, 2],
        c = [1, 1],
        n_firms = 2,
    )
    p_range_pad = ξ * (p_monop_opt - p_Bert_nash_equilibrium)
    price_options = [range(p_Bert_nash_equilibrium, p_monop_opt, n_prices)...]

    runCalvano(α, β, δ,
        price_options,
        competition_params,
        p_Bert_nash_equilibrium,
        p_monop_opt,
    )
end
