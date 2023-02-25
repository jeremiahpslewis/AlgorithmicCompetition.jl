using Test
using JuMP
using Chain
using AlgorithmicCompetition:
    AlgorithmicCompetition,
    CompetitionParameters,
    CompetitionParameters,
    CalvanoHyperParameters,
    CalvanoPolicy,
    CalvanoEnv,
    CompetitionSolution,
    ConvergenceCheck,
    solve_monopolist,
    solve_bertrand,
    p_BR,
    map_vect_to_int,
    map_int_to_vect,
    q_fun,
    run,
    run_and_extract

@testset "Competitive Equilibrium: Monopoly" begin
    params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])
    model_monop, p_monop = solve_monopolist(params)

    # symmetric solution found
    @test value(p_monop[1]) ≈ value(p_monop[2])

    # Match Calvano 2020 parameterization
    @test value(p_monop[1]) ≈ 1.92498 atol = 0.0001
    p_monop_opt = value(p_monop[2])
end

@testset "Competitive Equilibrium: Bertrand" begin
    params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])

    p_Bertrand_ = value.(solve_bertrand(params)[2])
    p_Bertrand = p_Bertrand_[1]

    # Parameter recovery
    @test p_Bertrand_[2] ≈ 1.47293 atol = 1e-3

    # Best response function matches Calvano 2020
    @test p_BR(1.47293, params) ≈ 1.47293 atol = 1e6
end

@testset "map_vect_to_int" begin
    n_prices = 15
    n_players = 2
    memory_length = 1
    n_state_space = n_prices^(n_players * memory_length)
    @test map_vect_to_int(repeat([n_prices], n_players), n_prices) - n_prices == n_state_space
    @test map_vect_to_int(Array{Int,2}(repeat([n_prices], n_players)'), n_prices) - n_prices ==
          n_state_space
end

@testset "run Calvano full simulation" begin
    p_monop_opt = 1.92498
    p_Bert_nash_equilibrium = 1.47

    α = Float32(0.125)
    β = Float32(1e-5)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = Int(1e4)
    price_index = 1:n_prices

    competition_params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])

    competition_solution = CompetitionSolution(competition_params)

    hyperparams = CalvanoHyperParameters(α, β, δ, max_iter, competition_solution; convergence_threshold=10)

    run([hyperparams]; stop_on_convergence=false)
    run(hyperparams; stop_on_convergence=false)
end

@testset "Run a set of experiments." begin
    competition_params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])

    competition_solution = CompetitionSolution(competition_params)

    n_increments = 10
    α_ = Float32.(range(0.025, 0.25, n_increments))
    β_ = Float32.(range(1.25e-8, 2e-5, n_increments))
    δ = 0.95
    max_iter = Int(1e4)

    hyperparameter_vect = [
        CalvanoHyperParameters(α, β, δ, max_iter, competition_solution; convergence_threshold=10) for α in α_ for β in β_
    ]

    exps = @chain hyperparameter_vect run_and_extract.(stop_on_convergence=true)
        
    # @test exps[1] isa Experiment
end

@testset "CompetitionParameters" begin
    @test CompetitionParameters(1, 1, [1.0, 1], [1.0, 1]) isa CompetitionParameters
    @test_throws DimensionMismatch CompetitionParameters(1, 1, [1.0, 1], [1.0])
end

@testset "q_fun" begin
    @test q_fun([1.47293, 1.47293], CompetitionParameters(0.25, 0, [2, 2], [1, 1])) ≈
          fill(0.47138, 2) atol = 0.01
    @test q_fun([1.92498, 1.92498], CompetitionParameters(0.25, 0, [2, 2], [1, 1])) ≈
          fill(0.36486, 2) atol = 0.01
end

@testset "Convergence Check Hook" begin
    competition_params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])
    competition_solution = CompetitionSolution(competition_params)

    env = CalvanoHyperParameters(Float32(0.1), Float32(1e-4), 0.95, Int(1e7), competition_solution) |> CalvanoEnv
    policies = env |> CalvanoPolicy
    # q_table = policies.agents[1].policy.policy.learner.approximator.table
    q_table = zeros(Float64, 15, 50625)

    approximator_table__state_argmax = zeros(UInt, env.n_players, env.n_state_space)
    prev_best_action = (@view approximator_table__state_argmax[1, :])[1]
    state = 20
end

@testset "map_vect_to_int, map_int_to_vect" begin
    vect_ = [1, 2, 3]
    base = 24
    i_num = map_vect_to_int(vect_, base)
    @test [vect_..., 0, 0] == map_int_to_vect(i_num, base, 5)

    int_ = 720
    vect_1 = map_int_to_vect(int_, base, 6)
    @test int_ == map_vect_to_int(vect_1, base)
end
