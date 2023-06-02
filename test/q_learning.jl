@testset "Q-Learning" begin
    n_prices = 15
    n_state_space = 15^2
    app = TabularApproximator(
        InitMatrix(n_prices, n_state_space),
        Descent(0.1),
    )

    @test Q(app, 1, 1) == 0
    @test Q(app, 1) == zeros(n_prices)
    @test 0.0625 == Q!(app, 1, 1, 1, 0.125, 0.5, 0.95)
end

