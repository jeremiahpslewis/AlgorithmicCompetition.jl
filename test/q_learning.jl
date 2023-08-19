@testset "Q-Learning" begin
    n_prices = 15
    n_state_space = 15^2
    α = Float64(0.125)
    β = Float64(1)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = 1000
    price_index = 1:n_prices

    competition_params_dict = Dict(
        :high => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
    )
    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

    hyperparameters = AIAPCHyperParameters(
        α,
        β,
        δ,
        max_iter,
        competition_solution_dict;
        convergence_threshold = 1,
    )
    env = AIAPCEnv(hyperparameters)

    app = TabularApproximator(InitMatrix(env; mode = "zero"), Descent(0.1))

    @test Q(app, 1, 1) == 0
    @test Q(app, 1) == zeros(n_prices)
    @test 0.0625 == Q!(app, 1, 1, 1, 0.125, 0.5, 0.95)
end

@testset "Q_i_0 AIAPC" begin
    α = Float64(0.125)
    β = Float64(1)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = 1000
    price_index = 1:n_prices

    competition_params_dict = Dict(
        :high => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
    )
    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

    hyperparameters = AIAPCHyperParameters(
        α,
        β,
        δ,
        max_iter,
        competition_solution_dict;
        convergence_threshold = 1,
    )
    env = AIAPCEnv(hyperparameters)

    test_prices = Q_i_0(env)

    @test minimum(test_prices) ≈ 4.111178690372623 atol = 0.001
    @test maximum(test_prices) ≈ 6.278004857861001 atol = 0.001
end

@testset "Q_i_0 DDDC" begin
    α = Float64(0.125)
    β = Float64(4e-1)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = Int(1e6) # 1e8
    price_index = 1:n_prices
    
    competition_params_dict = Dict(
        :high => CompetitionParameters(0.25, -0.25, (2, 2), (1, 1)),
        :low => CompetitionParameters(0.25, 0.25, (2, 2), (1, 1)),
    )
    
    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])
    
    data_demand_digital_params = DataDemandDigitalParams(
        low_signal_quality_level = 0.5,
        high_signal_quality_level = 0.5,
        signal_quality_is_high = [true, false],
        frequency_high_demand = 0.5,
    )
    
    hyperparams = DDDCHyperParameters(
        α,
        β,
        δ,
        max_iter,
        competition_solution_dict,
        data_demand_digital_params;
        convergence_threshold = Int(1e5),
    )
    
    
    env = DDDCEnv(hyperparams)

    @test minimum(Q_i_0(env)) == 0.2003206598478015
    @test maximum(Q_i_0(env)) == 0.3694013307458184
end

@testset "Q" begin
    params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))
    p_ = [1, 1]
    logit_demand = exp.((params.a .- p_) ./ params.μ)
    q_logit_demand =
        logit_demand /
        (sum(exp.((params.a .- p_) ./ params.μ)) + exp(params.a_0 / params.μ))

    Q(p_[1], p_[2], params) == q_logit_demand

    @test Q(1.47293, 1.47293, CompetitionParameters(0.25, 0, (2, 2), (1, 1))) ≈
          fill(0.47138, 2) atol = 0.01
    @test Q(1.92498, 1.92498, CompetitionParameters(0.25, 0, (2, 2), (1, 1))) ≈
          fill(0.36486, 2) atol = 0.01
end

@testset "simple InitMatrix test" begin
    α = Float64(0.125)
    β = Float64(1)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = 1000
    price_index = 1:n_prices

    competition_params_dict = Dict(
        :high => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
    )
    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

    hyperparameters = AIAPCHyperParameters(
        α,
        β,
        δ,
        max_iter,
        competition_solution_dict;
        convergence_threshold = 1,
    )
    env = AIAPCEnv(hyperparameters)

    a = InitMatrix(env; mode = "baseline")
    @test mean(a) ≈ 5.598115514452509
    @test a[1, 1] ≈ 5.7897603960172805
    @test a[1, 10] ≈ 5.7897603960172805
    @test a[5, 10] ≈ 6.278004857861001
end
