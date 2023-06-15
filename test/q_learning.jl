@testset "Q-Learning" begin
    n_prices = 15
    n_state_space = 15^2
    α = Float64(0.125)
    β = Float64(1e-5)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = 1000
    price_index = 1:n_prices

    competition_params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))

    competition_solution = CompetitionSolution(competition_params)

    hyperparameters = AIAPCHyperParameters(
        α,
        β,
        δ,
        max_iter,
        competition_solution;
        convergence_threshold = 1,
    )
    env = AIAPCEnv(hyperparameters)

    app = TabularApproximator(InitMatrix(env; mode = "zero"), Descent(0.1))

    @test Q(app, 1, 1) == 0
    @test Q(app, 1) == zeros(n_prices)
    @test 0.0625 == Q!(app, 1, 1, 1, 0.125, 0.5, 0.95)
end
