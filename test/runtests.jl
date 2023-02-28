using Test
using JuMP
using Chain
using ReinforcementLearning: PostActStage, state, reward, PostEpisodeStage, SequentialEnv, current_player, action_space, VectorSARTTrajectory, EpsilonGreedyExplorer
using ReinforcementLearningBase: test_interfaces!, test_runnable!
import ReinforcementLearningCore
using AlgorithmicCompetition:
    AlgorithmicCompetition,
    CompetitionParameters,
    CompetitionParameters,
    AIAPCHyperParameters,
    AIAPCPolicy,
    AIAPCEnv,
    CompetitionSolution,
    ConvergenceCheck,
    solve_monopolist,
    solve_bertrand,
    p_BR,
    construct_state_space_lookup,
    map_vect_to_int,
    map_int_to_vect,
    construct_profit_array,
    q_fun,
    run,
    run_and_extract,
    Experiment,
    reward,
    InitMatrix,
    get_ϵ,
    AIAPCEpsilonGreedyExplorer


@testset "Prepackaged Environment Tests" begin
    α = Float32(0.125)
    β = Float32(1e-5)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = 1000
    price_index = 1:n_prices

    competition_params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])

    competition_solution = CompetitionSolution(competition_params)

    hyperparams = AIAPCHyperParameters(α, β, δ, max_iter, competition_solution; convergence_threshold=1)
        
    test_interfaces!(AIAPCEnv(hyperparams))
    test_runnable!(AIAPCEnv(hyperparams))
end
@testset "Competitive Equilibrium: Monopoly" begin
    params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])
    model_monop, p_monop = solve_monopolist(params)

    # symmetric solution found
    @test value(p_monop[1]) ≈ value(p_monop[2])

    # Match AIAPC 2020 parameterization
    @test value(p_monop[1]) ≈ 1.92498 atol = 0.0001
    p_monop_opt = value(p_monop[2])
end

@testset "Competitive Equilibrium: Bertrand" begin
    params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])

    p_Bertrand_ = value.(solve_bertrand(params)[2])
    p_Bertrand = p_Bertrand_[1]

    # Parameter recovery
    @test p_Bertrand_[2] ≈ 1.47293 atol = 1e-3

    # Best response function matches AIAPC 2020
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

@testset "construct_state_space_lookup" begin
    @test construct_state_space_lookup(((1,1), (1,2), (2,1), (2,2)), 2) == [1 3; 2 4]
end

@testset "run AIAPC full simulation" begin
    α = Float32(0.125)
    β = Float32(1e-5)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = 1000
    price_index = 1:n_prices

    competition_params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])

    competition_solution = CompetitionSolution(competition_params)

    hyperparams = AIAPCHyperParameters(α, β, δ, max_iter, competition_solution; convergence_threshold=1)

    c_out = run(hyperparams; stop_on_convergence=false)
    
    # ensure that the policy is updated by the learner
    @test sum(c_out.policy.agents[1].policy.policy.learner.approximator.table .!= 0) != 0
    @test length(reward(c_out.env.env)) == 2
    @test length(reward(c_out.env.env, 1)) == 1

    c_out.env.env.is_done[1] = false
    @test reward(c_out.env.env) == [0, 0]
    @test reward(c_out.env.env, 1) != 0


    @test state(c_out.env) != 1
    @test sum(c_out.hook.hooks[1][2].best_response_vector == 0) == 0
    @test c_out.hook.hooks[1][2].best_response_vector != c_out.hook.hooks[2][2].best_response_vector

    run([hyperparams]; stop_on_convergence=false)
end

@testset "Run a set of experiments." begin
    competition_params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])

    competition_solution = CompetitionSolution(competition_params)

    n_increments = 3
    α_ = Float32.(range(0.025, 0.25, n_increments))
    β_ = Float32.(range(1.25e-8, 2e-5, n_increments))
    δ = 0.95
    max_iter = Int(1e4)

    hyperparameter_vect = [
        AIAPCHyperParameters(α, β, δ, max_iter, competition_solution; convergence_threshold=10) for α in α_ for β in β_
    ]

    expers = @chain hyperparameter_vect run_and_extract.(stop_on_convergence=true)
        
    # @test expers[1] isa Experiment
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

    env = AIAPCHyperParameters(Float32(0.1), Float32(1e-4), 0.95, Int(1e7), competition_solution) |> AIAPCEnv
    exper = Experiment(env)
    state(env)
    policies = env |> AIAPCPolicy
    AlgorithmicCompetition.update!(exper.hook.hooks[1][2], Int16(2), 3, false)
    @test exper.hook.hooks[1][2].best_response_vector[2] == 3

    
    policies[1].policy.policy.learner.approximator.table[11, :] .= 2
    exper.hook.hooks[1][2](PostEpisodeStage(), policies[1], exper.env)
    @test exper.hook.hooks[1][2].best_response_vector[state(env)] == 11
end

@testset "Profit array test" begin
    competition_params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])
    competition_solution = CompetitionSolution(competition_params)

    env = AIAPCHyperParameters(Float32(0.1), Float32(1e-4), 0.95, Int(1e7), competition_solution) |> AIAPCEnv
    exper = Experiment(env)

    price_options = env.price_options
    profit_function = env.profit_function
    action_space_ = env.action_space
    profit_array = construct_profit_array(action_space_, price_options, profit_function, 2)
    
    profit_array[5, 3, :] ≈ env.profit_function([price_options[5], price_options[3]])
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

@testset "simple InitMatrix test" begin
    a = InitMatrix(15, 225)
    b = InitMatrix(15, 225)
    a[1,1] = 10
    @test a[1,1] == 10
    @test b[1,1] == 0
end

@testset "Parameter / learning checks" begin
    α = Float32(0.125)
    β = Float32(1e-5)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = 1000
    price_index = 1:n_prices

    competition_params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])

    competition_solution = CompetitionSolution(competition_params)

    hyperparams = AIAPCHyperParameters(α, β, δ, max_iter, competition_solution; convergence_threshold=1)


    c_out = run(hyperparams; stop_on_convergence=false)

    # ensure that the policy is updated by the learner
    @test sum(c_out.policy.agents[1].policy.policy.learner.approximator.table .!= 0) != 0
    @test sum(c_out.policy.agents[2].policy.policy.learner.approximator.table .!= 0) != 0
    @test c_out.env.env.is_done[1]
    @test c_out.hook.hooks[1][2].iterations_until_convergence == max_iter
    @test c_out.hook.hooks[2][2].iterations_until_convergence == max_iter


    @test c_out.policy.agents[1].trajectory[:reward][1] .!= 0
    @test c_out.policy.agents[2].trajectory[:reward][1] .!= 0

    @test c_out.policy.agents[1].policy.policy.learner.approximator.table != c_out.policy.agents[2].policy.policy.learner.approximator.table
    @test c_out.hook.hooks[1][2].best_response_vector != c_out.hook.hooks[2][2].best_response_vector

    for i in 1:2
        @test c_out.hook[i][2].convergence_duration == 0
        @test !c_out.hook[i][2].is_converged
        @test c_out.hook[i][2].convergence_threshold == 1
        @test sum(c_out.hook[i][1].rewards .== 0) == 0
    end

    @test reward(c_out.env, 1) != 0
    @test reward(c_out.env, 2) != 0
    @test length(reward(c_out.env.env)) == 2
    @test length(c_out.env.env.action_space) == 225
    @test length(reward(c_out.env)) == 1


end

@testset "Sequential environment" begin
    α = Float32(0.125)
    β = Float32(1e-5)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = 1000
    price_index = 1:n_prices

    competition_params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])

    competition_solution = CompetitionSolution(competition_params)

    hyperparams = AIAPCHyperParameters(α, β, δ, max_iter, competition_solution; convergence_threshold=1)

    seq_env = AIAPCEnv(hyperparams) |> SequentialEnv
    @test current_player(seq_env) == 1
    @test action_space(seq_env) == 1:15
    @test reward(seq_env) != 0 # reward reflects outcomes of last play (which happens at player = 1, e.g. before any actions chosen)
    seq_env(5)
    @test current_player(seq_env) == 2
    @test reward(seq_env) == 0 # reward is zero as at least one player has already played (technically sequental plays)
end

@testset "Convergence stop works" begin
    α = Float32(0.125)
    β = Float32(1e-6)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = Int(1e6)
    price_index = 1:n_prices

    competition_params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])

    competition_solution = CompetitionSolution(competition_params)

    hyperparams = AIAPCHyperParameters(α, β, δ, max_iter, competition_solution; convergence_threshold=1)

    c_out = run(hyperparams; stop_on_convergence=false)
end

@testset "custom tdlearner update, to be upstreamed" begin
    t = VectorSARTTrajectory()
    l = TDLearner(TabularQApproximator(n_state=10, n_action=3), 0.5, :SARS, 1)
    _update!(l, l.approximator, Val{:SARS}(), t, PostEpisodeStage())
end

@testset "EpsilonGreedy" begin
    explorer = EpsilonGreedyExplorer(kind=:exp, ϵ_init=1, ϵ_stable=0, decay_steps = Int(round(1 / 1e-5)))
    @test ReinforcementLearningCore.get_ϵ(explorer, 1e5) ≈ 0.36787944117144233
    @test_broken ReinforcementLearningCore.get_ϵ(explorer, 1e5) ≈ 0.14 # Percentage cited in AIAPC paper

    explorer = AIAPCEpsilonGreedyExplorer(Float32(1e-5))
    @test get_ϵ(explorer, 1e5) ≈ 0.3678794504648588
    @test_broken get_ϵ(explorer, 1e5) ≈ 0.14 # Percentage cited in AIAPC paper
end
