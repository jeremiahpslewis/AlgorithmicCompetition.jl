using Test
using JuMP
using Chain
using ReinforcementLearningCore:
    PostActStage,
    state,
    reward,
    PostEpisodeStage,
    current_player,
    action_space,
    EpsilonGreedyExplorer,
    RandomPolicy,
    PreActStage,
    MultiAgentPolicy
using ReinforcementLearningBase: RLBase, test_interfaces!, test_runnable!, AbstractPolicy
import ReinforcementLearningCore
using StaticArrays
using Statistics
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
    AIAPCEpsilonGreedyExplorer,
    AIAPCSummary,
    TDLearner
using Distributed

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

    hyperparams = AIAPCHyperParameters(
        α,
        β,
        δ,
        max_iter,
        competition_solution;
        convergence_threshold = 1,
    )

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
    @test map_vect_to_int(repeat([n_prices], n_players), n_prices) - n_prices ==
          n_state_space
    @test map_vect_to_int(Array{Int,2}(repeat([n_prices], n_players)'), n_prices) -
          n_prices == n_state_space
end

@testset "construct_state_space_lookup" begin
    @test construct_state_space_lookup(((1, 1), (1, 2), (2, 1), (2, 2)), 2) == [1 3; 2 4]
end


@testset "Policy operation test" begin
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

    hyperparams = AIAPCHyperParameters(
        α,
        β,
        δ,
        max_iter,
        competition_solution;
        convergence_threshold = 1,
    )
    env = AIAPCEnv(hyperparams)
    policy = AIAPCPolicy(env)

    # Test full policy exploration of states
    policy(PreActStage(), env)
    n_ = Int(1e5)
    policy_runs = [[policy(env)...] for i in 1:n_]
    checksum_ = [sum(unique(policy_runs[j][i] for j in 1:n_)) for i in 1:2]
    @test all(checksum_ .== sum(1:env.n_prices))
end

@testset "run full AIAPC simulation" begin
    α = Float32(0.125)
    β = Float32(1e-5)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = Int(1e3)
    price_index = 1:n_prices

    competition_params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])

    competition_solution = CompetitionSolution(competition_params)

    hyperparams = AIAPCHyperParameters(
        α,
        β,
        δ,
        max_iter,
        competition_solution;
        convergence_threshold = 1,
    )

    c_out = run(hyperparams; stop_on_convergence = false)

    # ensure that the policy is updated by the learner
    @test sum(c_out.policy[Symbol(1)].policy.learner.approximator.table .!= 0) != 0
    @test length(reward(c_out.env)) == 2
    @test length(reward(c_out.env, 1)) == 1

    c_out.env.is_done[1] = false
    @test reward(c_out.env) == [0, 0]
    @test reward(c_out.env, 1) != 0


    @test state(c_out.env) != 1
    @test sum(c_out.hook[Symbol(1)][2].best_response_vector == 0) == 0
    @test c_out.hook[Symbol(1)][2].best_response_vector !=
          c_out.hook[Symbol(2)][2].best_response_vector
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
        AIAPCHyperParameters(
            α,
            β,
            δ,
            max_iter,
            competition_solution;
            convergence_threshold = 10,
        ) for α in α_ for β in β_
    ]

    experiments = @chain hyperparameter_vect run_and_extract.(stop_on_convergence = true)

    @test experiments[1] isa AIAPCSummary
    @test 10 < experiments[1].iterations_until_convergence < max_iter  
    # @test (sum(experiments[1].avg_profit .> 1) + sum(experiments[1].avg_profit .< 0)) == 0
    # @test experiments[1].avg_profit[1] != experiments[1].avg_profit[2]
    @test all(experiments[1].is_converged)
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

    env =
        AIAPCHyperParameters(
            Float32(0.1),
            Float32(1e-4),
            0.95,
            Int(1e7),
            competition_solution,
        ) |> AIAPCEnv
    exper = Experiment(env)
    state(env)
    policies = env |> AIAPCPolicy
    exper.hook[Symbol(1)][2](Int16(2), 3, false)
    @test exper.hook[Symbol(1)][2].best_response_vector[2] == 3


    policies[Symbol(1)].policy.learner.approximator.table[11, :] .= 2
    exper.hook[Symbol(1)][2](PostEpisodeStage(), policies[Symbol(1)], exper.env, :p1)
    @test exper.hook[Symbol(1)][2].best_response_vector[state(env)] == 11
end

@testset "Profit array test" begin
    competition_params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])
    competition_solution = CompetitionSolution(competition_params)

    env =
        AIAPCHyperParameters(
            Float32(0.1),
            Float32(1e-4),
            0.95,
            Int(1e7),
            competition_solution,
        ) |> AIAPCEnv
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
    a[1, 1] = 10
    @test a[1, 1] == 10
    @test b[1, 1] == 0
end

@testset "Parameter / learning checks" begin
    α = Float32(0.125)
    β = Float32(1e-5)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = 5000
    price_index = 1:n_prices

    competition_params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])

    competition_solution = CompetitionSolution(competition_params)

    hyperparams = AIAPCHyperParameters(
        α,
        β,
        δ,
        max_iter,
        competition_solution;
        convergence_threshold = 1,
    )


    c_out = run(hyperparams; stop_on_convergence = false)

    # ensure that the policy is updated by the learner
    @test sum(c_out.policy[Symbol(1)].policy.learner.approximator.table .!= 0) != 0
    @test sum(c_out.policy[Symbol(2)].policy.learner.approximator.table .!= 0) != 0
    @test c_out.env.is_done[1]
    @test c_out.hook[Symbol(1)][2].iterations_until_convergence == max_iter
    @test c_out.hook[Symbol(2)][2].iterations_until_convergence == max_iter


    @test c_out.policy[Symbol(1)].trajectory.container[:reward][1] .!= 0
    @test c_out.policy[Symbol(2)].trajectory.container[:reward][1] .!= 0

    @test c_out.policy[Symbol(1)].policy.learner.approximator.table !=
          c_out.policy[Symbol(2)].policy.learner.approximator.table
    @test c_out.hook[Symbol(1)][2].best_response_vector !=
          c_out.hook[Symbol(2)][2].best_response_vector


    # @test mean(
    #     c_out.hook[Symbol(1)][1].rewards[(end-2):end] .!=
    #     c_out.hook[Symbol(2)][1].rewards[(end-2):end],
    # ) >= 0.3

    for i = [Symbol(1), Symbol(2)]
        @test c_out.hook[i][2].convergence_duration >= 0
        @test c_out.hook[i][2].is_converged
        @test c_out.hook[i][2].convergence_threshold == 1
        # @test sum(c_out.hook[i][1].rewards .== 0) == 0
    end

    @test reward(c_out.env, 1) != 0
    @test reward(c_out.env, 2) != 0
    @test length(reward(c_out.env)) == 2
    @test length(c_out.env.action_space) == 225
    # @test length(reward(c_out.env)) == 1


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

    hyperparams = AIAPCHyperParameters(
        α,
        β,
        δ,
        max_iter,
        competition_solution;
        convergence_threshold = 1,
    )

    env = AIAPCEnv(hyperparams)
    @test current_player(env) == RLBase.SimultaneousPlayer()
    @test action_space(env, Symbol(1)) == 1:15
    @test reward(env) != 0 # reward reflects outcomes of last play (which happens at player = 1, e.g. before any actions chosen)
    env((5, 5))
    @test reward(env) != [0,0] # reward is zero as at least one player has already played (technically sequental plays)
end

@testset "Convergence stop works" begin
    α = Float32(0.125)
    β = Float32(1e-5)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = Int(1e6)
    price_index = 1:n_prices

    competition_params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])

    competition_solution = CompetitionSolution(competition_params)

    hyperparams = AIAPCHyperParameters(
        α,
        β,
        δ,
        max_iter,
        competition_solution;
        convergence_threshold = 10,
    )
    c_out = run(hyperparams; stop_on_convergence = false)
    @test get_ϵ(c_out.policy[Symbol(1)].policy.explorer) < 1e-4
    @test get_ϵ(c_out.policy[Symbol(2)].policy.explorer) < 1e-4

    max_iter = Int(1e7)
    hyperparams = AIAPCHyperParameters(
        α,
        β,
        δ,
        max_iter,
        competition_solution;
        convergence_threshold = 10,
    )
    c_out = run(hyperparams; stop_on_convergence = true)
    @test 0.98 < get_ϵ(c_out.policy[Symbol(1)].policy.explorer) < 1
    @test 0.98 < get_ϵ(c_out.policy[Symbol(2)].policy.explorer) < 1

    @test c_out.stop_condition.stop_conditions[1](1, c_out.env) == false
    @test c_out.stop_condition.stop_conditions[2](1, c_out.env) == true
    
    @test_broken c_out.hook[Symbol(2)][2].convergence_duration == 10
    @test c_out.hook[Symbol(2)][2].convergence_duration >= 0
    @test c_out.hook[Symbol(1)][2].convergence_duration >= 0
    @test c_out.env.convergence_dict[:1] < max_iter
end

@testset "EpsilonGreedy" begin
    explorer = EpsilonGreedyExplorer(
        kind = :exp,
        ϵ_init = 1,
        ϵ_stable = 0,
        decay_steps = Int(round(1 / 1e-5)),
    )
    @test ReinforcementLearningCore.get_ϵ(explorer, 1e5) ≈ 0.36787944117144233
    @test_broken ReinforcementLearningCore.get_ϵ(explorer, 1e5) ≈ 0.14 # Percentage cited in AIAPC paper

    explorer = AIAPCEpsilonGreedyExplorer(Float32(1e-5))
    @test get_ϵ(explorer, 1e5) ≈ 0.3678794504648588
    @test_broken get_ϵ(explorer, 1e5) ≈ 0.14 # Percentage cited in AIAPC paper
end

@testset "ConvergenceCheck" begin
    competition_params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])
    competition_solution = CompetitionSolution(competition_params)

    env =
        AIAPCHyperParameters(
            Float32(0.1),
            Float32(1e-4),
            0.95,
            Int(1e7),
            competition_solution,
        ) |> AIAPCEnv
    policies = env |> AIAPCPolicy

    convergence_hook = ConvergenceCheck(1)
    convergence_hook(PostEpisodeStage(), policies[Symbol(1)], env, :player_1)
    @test convergence_hook.convergence_duration == 0
    @test convergence_hook.iterations_until_convergence == 1
    @test convergence_hook.best_response_vector[1] == 1
    @test convergence_hook.is_converged != true

    convergence_hook_1 = ConvergenceCheck(1)
    convergence_hook_1.best_response_vector = MVector{225,Int}(fill(1, 225))
    convergence_hook_1(PostEpisodeStage(), policies[Symbol(1)], env, :player_1)

    @test convergence_hook.iterations_until_convergence == 1
    @test convergence_hook.convergence_duration ∈ [0, 1]
    @test convergence_hook_1.is_converged == true
end

@testset "run multiprocessing code" begin
    n_procs_ = 1

    _procs = addprocs(
        n_procs_,
        topology = :master_worker,
        exeflags = ["--threads=1", "--project=$(Base.active_project())"],
    )

    @everywhere begin
        using Pkg
        Pkg.instantiate()
        using AlgorithmicCompetition
    end

    AlgorithmicCompetition.run_aiapc(;
        n_parameter_iterations = 1,
        max_iter = Int(100),
        convergence_threshold = Int(10),
    )

    rmprocs(_procs)
end
