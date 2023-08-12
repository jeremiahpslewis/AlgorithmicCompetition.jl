@testset "Prepackaged Environment Tests" begin
    α = Float64(0.125)
    β = Float64(1)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = 1000

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

    # Until state handling is fixed for multi-agent simultaneous environments, we can't test this
    # @test test_interfaces!(AIAPCEnv(hyperparameters))
    # @test test_runnable!(AIAPCEnv(hyperparameters))
end

@testset "Profit gain check" begin
    competition_params_dict = Dict(
        :high => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
    )
    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

    env =
        AIAPCHyperParameters(
            Float64(0.1),
            Float64(1e-4),
            0.95,
            Int(1e7),
            competition_solution_dict,
        ) |> AIAPCEnv
    env.memory[1] = CartesianIndex(Int8(1), Int8(1))
    exper = Experiment(env; debug = true)

    # Find the Nash equilibrium profit
    params = env.competition_params_dict[:high]
    p_Bert_nash_equilibrium = exper.env.p_Bert_nash_equilibrium
    π_min_price =
        π(minimum(exper.env.price_options), minimum(exper.env.price_options), params)[1]

    π_nash = π(p_Bert_nash_equilibrium, p_Bert_nash_equilibrium, params)[1]
    @test π_nash > π_min_price
    for i = 1:exper.hook[Symbol(1)].hooks[2].rewards.capacity
        push!(exper.hook[Symbol(1)].hooks[2].rewards, π_nash)
        push!(exper.hook[Symbol(2)].hooks[2].rewards, 0)
    end

    ec_summary_ = economic_summary(exper)
    # @test round(profit_gain(ec_summary_.convergence_profit[1], env); digits = 2) == 0
    # @test round(profit_gain(ec_summary_.convergence_profit[2], env); digits = 2) == 1.07

    p_monop_opt = exper.env.p_monop_opt
    π_monop = π(p_monop_opt, p_monop_opt, params)[1]
    π_max_price =
        π(maximum(exper.env.price_options), maximum(exper.env.price_options), params)[1]
    @test π_max_price < π_monop

    for i = 1:exper.hook[Symbol(1)].hooks[2].rewards.capacity
        push!(exper.hook[Symbol(1)].hooks[2].rewards, π_monop)
        push!(exper.hook[Symbol(2)].hooks[2].rewards, 0)
    end

    ec_summary_ = economic_summary(exper)
    @test 1 > round(profit_gain(ec_summary_.convergence_profit[1], env); digits = 2) > 0
end

@testset "Sequential environment" begin
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
    @test current_player(env) == RLBase.SimultaneousPlayer()
    @test action_space(env, Symbol(1)) == Int8.(1:15)
    @test reward(env) != 0 # reward reflects outcomes of last play (which happens at player = 1, e.g. before any actions chosen)
    act!(env, CartesianIndex(Int8(5), Int8(5)))
    @test reward(env) != [0, 0] # reward is zero as at least one player has already played (technically sequental plays)
end

@testset "run AIAPC multiprocessing code" begin
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

@testset "run DDDC multiprocessing code" begin
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

    AlgorithmicCompetition.run_dddc(
        n_parameter_iterations = 1,
        max_iter = 10,
        convergence_threshold = 1,
        n_grid_increments = 3,
    )

    rmprocs(_procs)
end

@testset "run full AIAPC simulation (with full convergence threshold)" begin
    α = Float64(0.075)
    β = Float64(0.25)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = Int(1e9)
    price_index = 1:n_prices

    competition_params_dict = Dict(
        :high => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
    )
    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

    hyperparameters = AIAPCHyperParameters(α, β, δ, max_iter, competition_solution_dict)

    profit_gain_max = 0
    i = 0
    while (profit_gain_max <= 0.82) && (i < 10)
        i += 1
        c_out = run(hyperparameters; stop_on_convergence = true)
        profit_gain_max =
            maximum(profit_gain(economic_summary(c_out).convergence_profit, c_out.env))
    end
    @test profit_gain_max > 0.82

end

@testset "run full AIAPC simulation" begin
    α = Float64(0.125)
    β = Float64(4e-1)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = Int(1e6)
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
        convergence_threshold = 10000,
    )

    c_out = run(hyperparameters; stop_on_convergence = true)
    @test minimum(c_out.policy[Symbol(1)].policy.learner.approximator.table) < 6
    @test maximum(c_out.policy[Symbol(1)].policy.learner.approximator.table) > 5.5

    # ensure that the policy is updated by the learner
    @test sum(c_out.policy[Symbol(1)].policy.learner.approximator.table; dims = 2) != 0
    state_sum = sum(c_out.policy[Symbol(1)].policy.learner.approximator.table; dims = 1)
    @test !all(y -> y == state_sum[1], state_sum)
    @test length(reward(c_out.env)) == 2
    @test length(reward(c_out.env, 1)) == 1

    c_out.env.is_done[1] = false
    @test reward(c_out.env) == (0, 0)
    @test reward(c_out.env, 1) != 0

    @test sum(c_out.hook[Symbol(1)][1].best_response_vector == 0) == 0
    @test c_out.hook[Symbol(1)][1].best_response_vector !=
          c_out.hook[Symbol(2)][1].best_response_vector
end

@testset "Run a set of experiments." begin
    competition_params_dict = Dict(
        :high => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
    )
    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

    n_parameter_increments = 3
    α_ = Float64.(range(0.0025, 0.25, n_parameter_increments))
    β_ = Float64.(range(0.025, 2, n_parameter_increments))
    δ = 0.95
    max_iter = Int(1e8)

    hyperparameter_vect = [
        AIAPCHyperParameters(
            α,
            β,
            δ,
            max_iter,
            competition_solution_dict;
            convergence_threshold = 10,
        ) for α in α_ for β in β_
    ]

    experiments = @chain hyperparameter_vect run_and_extract.(stop_on_convergence = true)

    @test experiments[1] isa AIAPCSummary
    @test all(10 < experiments[1].iterations_until_convergence[i] < max_iter for i = 1:2)
    @test (
        sum(experiments[1].convergence_profit .> 1) +
        sum(experiments[1].convergence_profit .< 0)
    ) == 0
    # @test_broken experiments[1].convergence_profit[1] != experiments[1].convergence_profit[2]
    @test all(experiments[1].is_converged)
end

@testset "Parameter / learning checks" begin
    α = Float64(0.125)
    β = Float64(1)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = 10000
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


    c_out = run(hyperparameters; stop_on_convergence = false, debug = true)

    # ensure that the policy is updated by the learner
    @test sum(c_out.policy[Symbol(1)].policy.learner.approximator.table .!= 0) != 0
    @test sum(c_out.policy[Symbol(2)].policy.learner.approximator.table .!= 0) != 0
    @test c_out.env.is_done[1]
    @test c_out.hook[Symbol(1)][1].iterations_until_convergence == max_iter
    @test c_out.hook[Symbol(2)][1].iterations_until_convergence == max_iter


    @test c_out.policy[Symbol(1)].trajectory.container[:reward][1] .!= 0
    @test c_out.policy[Symbol(2)].trajectory.container[:reward][1] .!= 0

    @test c_out.policy[Symbol(1)].policy.learner.approximator.table !=
          c_out.policy[Symbol(2)].policy.learner.approximator.table
    @test c_out.hook[Symbol(1)][1].best_response_vector !=
          c_out.hook[Symbol(2)][1].best_response_vector


    @test mean(
        c_out.hook[Symbol(1)][2].rewards[(end-2):end] .!=
        c_out.hook[Symbol(2)][2].rewards[(end-2):end],
    ) >= 0.3

    for i in [Symbol(1), Symbol(2)]
        @test c_out.hook[i][1].convergence_duration >= 0
        @test c_out.hook[i][1].is_converged
        @test c_out.hook[i][1].convergence_threshold == 1
        @test sum(c_out.hook[i][2].rewards .== 0) == 0
    end

    @test reward(c_out.env, 1) != 0
    @test reward(c_out.env, 2) != 0
    @test length(reward(c_out.env)) == 2
    @test length(c_out.env.action_space) == 225
    @test length(reward(c_out.env)) == 2
end

@testset "No stop on Convergence stop works" begin
    α = Float64(0.125)
    β = Float64(1)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = Int(1e6)
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
        convergence_threshold = 10,
    )
    c_out = run(hyperparameters; stop_on_convergence = false)
    @test get_ϵ(c_out.policy[Symbol(1)].policy.explorer) < 1e-4
    @test get_ϵ(c_out.policy[Symbol(2)].policy.explorer) < 1e-4
end

@testset "Convergence stop works" begin
    α = Float64(0.125)
    β = Float64(1)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = Int(1e7)
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
        convergence_threshold = 5,
    )
    c_out = run(hyperparameters; stop_on_convergence = true)
    @test 0.98 < get_ϵ(c_out.policy[Symbol(1)].policy.explorer) < 1
    @test 0.98 < get_ϵ(c_out.policy[Symbol(2)].policy.explorer) < 1

    @test RLCore.check_stop(c_out.stop_condition, 1, c_out.env) == true
    @test RLCore.check_stop(c_out.stop_condition.stop_conditions[1], 1, c_out.env) == false
    @test RLCore.check_stop(c_out.stop_condition.stop_conditions[2], 1, c_out.env) == true

    @test c_out.hook[Symbol(1)][1].convergence_duration >= 5
    @test c_out.hook[Symbol(2)][1].convergence_duration >= 5
    @test (c_out.hook[Symbol(2)][1].convergence_duration == 5) ||
          (c_out.hook[Symbol(1)][1].convergence_duration == 5)
end
