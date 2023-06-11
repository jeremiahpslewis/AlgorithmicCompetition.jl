using Test
using JuMP
using Chain
using ReinforcementLearningCore:
    RLCore,
    PostActStage,
    PreActStage,
    PostEpisodeStage,
    PreEpisodeStage,
    state,
    reward,
    current_player,
    action_space,
    EpsilonGreedyExplorer,
    RandomPolicy,
    MultiAgentPolicy,
    optimise!
using ReinforcementLearningBase:
    RLBase, test_interfaces!, test_runnable!, AbstractPolicy, act!, plan!
import ReinforcementLearningCore: RLCore
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
    Q,
    Q!,
    Q_i_0,
    π,
    run,
    run_and_extract,
    Experiment,
    reward,
    InitMatrix,
    get_ϵ,
    AIAPCEpsilonGreedyExplorer,
    AIAPCSummary,
    TDLearner,
    economic_summary,
    profit_gain
using Distributed

@testset "Prepackaged Environment Tests" begin
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

    test_interfaces!(AIAPCEnv(hyperparameters))
    test_runnable!(AIAPCEnv(hyperparameters))
end
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
    policy = AIAPCPolicy(env)

    # Test full policy exploration of states
    push!(policy, PreActStage(), env)
    n_ = Int(1e5)
    policy_runs = [[plan!(policy, env)...] for i = 1:n_]
    checksum_ = [sum(unique(policy_runs[j][i] for j = 1:n_)) for i = 1:2]
    @test all(checksum_ .== sum(1:env.n_prices))
end

@testset "Q_i_0" begin
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

    test_prices = Q_i_0(env.price_options, env.δ, env.competition_solution.params)
    @test minimum(test_prices) == 4.070906600373279
    @test maximum(test_prices) == 4.61282553058293
end

@testset "policy push! and optimise! test" begin
    competition_params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))
    competition_solution = CompetitionSolution(competition_params)

    env =
        AIAPCHyperParameters(
            Float64(0.1),
            Float64(1e-4),
            0.95,
            Int(1e7),
            competition_solution,
        ) |> AIAPCEnv

    policy = AIAPCPolicy(env)
    approx_table = copy(policy.agents[Symbol(1)].policy.learner.approximator.table)
    # First three rounds

    # t=1
    push!(policy, PreEpisodeStage(), env)
    push!(policy, PreActStage(), env)
    @test policy.agents[Symbol(1)].cache.reward == nothing
    @test policy.agents[Symbol(1)].cache.terminal == nothing
    @test length(policy.agents[Symbol(1)].trajectory.container) == 0
    optimise!(policy, PreActStage())
    approx_table_t_1 = copy(policy.agents[Symbol(1)].policy.learner.approximator.table)
    @test approx_table_t_1 == approx_table # test that optimise! in t=1 is a noop
    action = RLBase.plan!(policy, env)
    act!(env, action)
    @test length(policy.agents[Symbol(1)].trajectory.container) == 0 # test that trajectory has not been filled
    push!(policy, PostActStage(), env)
    reward_1 = copy(policy.agents[Symbol(1)].cache.reward)
    @test reward_1 != 0
    push!(policy, PostEpisodeStage(), env)
    cache_1 = deepcopy(policy.agents[Symbol(1)].cache)

    # t=2
    push!(policy, PreEpisodeStage(), env)
    push!(policy, PreActStage(), env)
    @test length(policy.agents[Symbol(1)].trajectory.container) == 1
    optimise!(policy, PreActStage())
    approx_table_t_2 = copy(policy.agents[Symbol(1)].policy.learner.approximator.table)
    @test approx_table_t_2 != approx_table_t_1 # test that optimise! in t=2 is not a noop   
    action = RLBase.plan!(policy, env)
    act!(env, action)
    push!(policy, PostActStage(), env)
    reward_2 = copy(policy.agents[Symbol(1)].cache.reward)
    @test reward_2 != reward_1
    push!(policy, PostEpisodeStage(), env)
    cache_2 = deepcopy(policy.agents[Symbol(1)].cache)
    @test cache_1.action != cache_2.action

    # t=3
    push!(policy, PreEpisodeStage(), env)
    push!(policy, PreActStage(), env)
    @test length(policy.agents[Symbol(1)].trajectory.container) == 1
    optimise!(policy, PreActStage())
    approx_table_t_3 = copy(policy.agents[Symbol(1)].policy.learner.approximator.table)
    @test approx_table_t_2 != approx_table_t_3 # test that optimise! in t=2 is not a noop
    action = RLBase.plan!(policy, env)
    act!(env, action)
    push!(policy, PostActStage(), env)
    reward_3 = copy(policy.agents[Symbol(1)].cache.reward)
    @test reward_3 != reward_2
    push!(policy, PostEpisodeStage(), env)
    cache_3 = deepcopy(policy.agents[Symbol(1)].cache)
    @test cache_2.action != cache_3.action
end

@testset "run full AIAPC simulation (with full convergence threshold)" begin
    α = Float64(0.075)
    β = Float64(0.25e-5)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = Int(1e9)
    price_index = 1:n_prices

    competition_params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))

    competition_solution = CompetitionSolution(competition_params)

    hyperparameters = AIAPCHyperParameters(
        α,
        β,
        δ,
        max_iter,
        competition_solution
    )

    profit_gain_max = 0
    i = 0
    while profit_gain_max <= 0.82 && i < 5
        i += 1
        c_out = run(hyperparameters; stop_on_convergence = true)
        profit_gain_max = maximum(profit_gain.(economic_summary(c_out).avg_profit, c_out.env))
    end
    @test profit_gain_max > 0.82

end

@testset "run full AIAPC simulation" begin
    α = Float64(0.125)
    β = Float64(1e-5)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = Int(1e5)
    price_index = 1:n_prices

    competition_params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))

    competition_solution = CompetitionSolution(competition_params)

    hyperparameters = AIAPCHyperParameters(
        α,
        β,
        δ,
        max_iter,
        competition_solution;
        convergence_threshold = 10,
    )

    c_out = run(hyperparameters; stop_on_convergence = true)
    # ensure that the policy is updated by the learner
    @test sum(c_out.policy[Symbol(1)].policy.learner.approximator.table; dims = 2) != 0
    state_sum = sum(c_out.policy[Symbol(1)].policy.learner.approximator.table; dims = 1)
    @test !all(y -> y == state_sum[1], state_sum)
    @test length(reward(c_out.env)) == 2
    @test length(reward(c_out.env, 1)) == 1

    c_out.env.is_done[1] = false
    @test reward(c_out.env) == [0, 0]
    @test reward(c_out.env, 1) != 0

    @test sum(c_out.hook[Symbol(1)][2].best_response_vector == 0) == 0
    @test c_out.hook[Symbol(1)][2].best_response_vector !=
          c_out.hook[Symbol(2)][2].best_response_vector
end

@testset "Run a set of experiments." begin
    competition_params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))

    competition_solution = CompetitionSolution(competition_params)

    n_parameter_increments = 3
    α_ = Float64.(range(0.0025, 0.25, n_parameter_increments))
    β_ = Float64.(range(0.025, 2, n_parameter_increments)) * 1e-5
    δ = 0.95
    max_iter = Int(1e8)

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
    @test all(10 < experiments[1].iterations_until_convergence[i] < max_iter for i = 1:2)
    @test (sum(experiments[1].avg_profit .> 1) + sum(experiments[1].avg_profit .< 0)) == 0
    @test experiments[1].avg_profit[1] != experiments[1].avg_profit[2]
    @test all(experiments[1].is_converged)
end

@testset "Profit gain check" begin
    competition_params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))
    competition_solution = CompetitionSolution(competition_params)

    env =
        AIAPCHyperParameters(
            Float64(0.1),
            Float64(1e-4),
            0.95,
            Int(1e7),
            competition_solution,
        ) |> AIAPCEnv
    exper = Experiment(env)

    # Find the Nash equilibrium profit
    params = env.competition_solution.params
    p_Bert_nash_equilibrium = exper.env.p_Bert_nash_equilibrium
    π_min_price =  π(minimum(exper.env.price_options), minimum(exper.env.price_options), params)[1]
    
    π_nash =  π(p_Bert_nash_equilibrium, p_Bert_nash_equilibrium, params)[1]
    @test π_nash > π_min_price
    for i in 1:exper.hook[Symbol(1)].hooks[1].rewards.capacity
        push!(exper.hook[Symbol(1)].hooks[1].rewards, π_nash)
        push!(exper.hook[Symbol(2)].hooks[1].rewards, 0)
    end

    ec_summary_ = economic_summary(exper)
    @test round(profit_gain(ec_summary_.avg_profit[1], env); digits=2) == 0
    @test round(profit_gain(ec_summary_.avg_profit[2], env); digits=2) < 0

    p_monop_opt = exper.env.p_monop_opt
    π_monop =  π(p_monop_opt, p_monop_opt, params)[1]
    π_max_price =  π(maximum(exper.env.price_options), maximum(exper.env.price_options), params)[1]
    @test π_max_price < π_monop

    for i in 1:exper.hook[Symbol(1)].hooks[1].rewards.capacity
        push!(exper.hook[Symbol(1)].hooks[1].rewards, π_monop)
        push!(exper.hook[Symbol(2)].hooks[1].rewards, 0)
    end

    ec_summary_ = economic_summary(exper)
    @test round(profit_gain(ec_summary_.avg_profit[1], env); digits=2) == 1
    @test round(profit_gain(ec_summary_.avg_profit[2], env); digits=2) < 0
end

@testset "CompetitionParameters" begin
    @test CompetitionParameters(1, 1, (1.0, 1.0), (1.0, 1.0)) isa CompetitionParameters
    @test_throws DimensionMismatch CompetitionParameters(1, 1, (1.0, 1), (1.0))
end

@testset "Q" begin
    @test Q(1.47293, 1.47293, CompetitionParameters(0.25, 0, (2, 2), (1, 1))) ≈
          fill(0.47138, 2) atol = 0.01
    @test Q(1.92498, 1.92498, CompetitionParameters(0.25, 0, (2, 2), (1, 1))) ≈
          fill(0.36486, 2) atol = 0.01
end

@testset "Convergence Check Hook" begin
    competition_params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))
    competition_solution = CompetitionSolution(competition_params)

    env =
        AIAPCHyperParameters(
            Float64(0.1),
            Float64(1e-4),
            0.95,
            Int(1e7),
            competition_solution,
        ) |> AIAPCEnv
    exper = Experiment(env)
    state(env)
    policies = env |> AIAPCPolicy
    push!(exper.hook[Symbol(1)][2], Int64(2), 3, false)
    @test exper.hook[Symbol(1)][2].best_response_vector[2] == 3


    policies[Symbol(1)].policy.learner.approximator.table[11, :] .= 2
    push!(exper.hook[Symbol(1)][2], PostActStage(), policies[Symbol(1)], exper.env, :p1)
    @test exper.hook[Symbol(1)][2].best_response_vector[state(env)] == 8
end

@testset "Profit array test" begin
    competition_params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))
    competition_solution = CompetitionSolution(competition_params)
    params = AIAPCHyperParameters(
        Float64(0.1),
        Float64(1e-4),
        0.95,
        Int(1e7),
        competition_solution,
    )
    env = params |> AIAPCEnv
    exper = Experiment(env)

    price_options = env.price_options
    action_space_ = env.action_space
    profit_array =
        construct_profit_array(action_space_, price_options, competition_solution.params, 2)

    profit_array[5, 3, :] ≈
    π(price_options[5], price_options[3], competition_solution.params)
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

    a = InitMatrix(
        env.price_options,
        env.n_state_space,
        env.δ,
        env.competition_solution.params,
        mode = "baseline",
    )
    @test a[1, 1] ≈ 4.070906600373279
    @test a[1, 10] ≈ 4.070906600373279
    @test a[5, 10] ≈ 4.521381412143576
end

@testset "Parameter / learning checks" begin
    α = Float64(0.125)
    β = Float64(1e-5)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = 10000
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


    c_out = run(hyperparameters; stop_on_convergence = false)

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


    @test mean(
        c_out.hook[Symbol(1)][1].rewards[(end-2):end] .!=
        c_out.hook[Symbol(2)][1].rewards[(end-2):end],
    ) >= 0.3

    for i in [Symbol(1), Symbol(2)]
        @test c_out.hook[i][2].convergence_duration >= 0
        @test c_out.hook[i][2].is_converged
        @test c_out.hook[i][2].convergence_threshold == 1
        @test sum(c_out.hook[i][1].rewards .== 0) == 0
    end

    @test reward(c_out.env, 1) != 0
    @test reward(c_out.env, 2) != 0
    @test length(reward(c_out.env)) == 2
    @test length(c_out.env.action_space) == 225
    @test length(reward(c_out.env)) == 2
end

@testset "Sequential environment" begin
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
    @test current_player(env) == RLBase.SimultaneousPlayer()
    @test action_space(env, Symbol(1)) == 1:15
    @test reward(env) != 0 # reward reflects outcomes of last play (which happens at player = 1, e.g. before any actions chosen)
    act!(env, (5, 5))
    @test reward(env) != [0, 0] # reward is zero as at least one player has already played (technically sequental plays)
end

@testset "No stop on Convergence stop works" begin
    α = Float64(0.125)
    β = Float64(1e-5)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = Int(1e6)
    price_index = 1:n_prices

    competition_params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))
    competition_solution = CompetitionSolution(competition_params)

    hyperparameters = AIAPCHyperParameters(
        α,
        β,
        δ,
        max_iter,
        competition_solution;
        convergence_threshold = 10,
    )
    c_out = run(hyperparameters; stop_on_convergence = false)
    @test get_ϵ(c_out.policy[Symbol(1)].policy.explorer) < 1e-4
    @test get_ϵ(c_out.policy[Symbol(2)].policy.explorer) < 1e-4
end

@testset "Convergence stop works" begin
    α = Float64(0.125)
    β = Float64(1e-5)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = Int(1e7)
    price_index = 1:n_prices

    competition_params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))
    competition_solution = CompetitionSolution(competition_params)

    hyperparameters = AIAPCHyperParameters(
        α,
        β,
        δ,
        max_iter,
        competition_solution;
        convergence_threshold = 5,
    )
    c_out = run(hyperparameters; stop_on_convergence = true)
    @test 0.98 < get_ϵ(c_out.policy[Symbol(1)].policy.explorer) < 1
    @test 0.98 < get_ϵ(c_out.policy[Symbol(2)].policy.explorer) < 1

    @test RLCore.check_stop(c_out.stop_condition, 1, c_out.env) == true
    @test RLCore.check_stop(c_out.stop_condition.stop_conditions[1], 1, c_out.env) == false
    @test RLCore.check_stop(c_out.stop_condition.stop_conditions[2], 1, c_out.env) == true

    @test c_out.hook[Symbol(1)][2].convergence_duration >= 5
    @test c_out.hook[Symbol(2)][2].convergence_duration >= 5
    @test (c_out.hook[Symbol(2)][2].convergence_duration == 5) ||
          (c_out.hook[Symbol(1)][2].convergence_duration == 5)
end

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

@testset "ConvergenceCheck" begin
    competition_params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))
    competition_solution = CompetitionSolution(competition_params)

    env =
        AIAPCHyperParameters(
            Float64(0.1),
            Float64(1e-4),
            0.95,
            Int(1e7),
            competition_solution,
        ) |> AIAPCEnv
    policies = env |> AIAPCPolicy

    convergence_hook = ConvergenceCheck(1)
    push!(convergence_hook, PostActStage(), policies[Symbol(1)], env, Symbol(1))
    @test convergence_hook.convergence_duration == 0
    @test convergence_hook.iterations_until_convergence == 1
    @test convergence_hook.best_response_vector[state(env, :1)] != 0
    @test convergence_hook.is_converged != true

    convergence_hook_1 = ConvergenceCheck(1)
    convergence_hook_1.best_response_vector = MVector{225,Int}(fill(8, 225))
    push!(convergence_hook_1, PostActStage(), policies[Symbol(1)], env, Symbol(1))

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
