using CircularArrayBuffers: capacity

@testset "Prepackaged Environment Tests" begin
    α = Float64(0.125)
    β = Float64(1)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = 10000

    competition_params_dict = Dict(
        :high => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
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

    RLBase.test_interfaces!(AIAPCEnv(hyperparameters))
    # Until state handling is fixed for multi-agent simultaneous environments, we can't test this
    # RLBase.test_runnable!(AIAPCEnv(hyperparameters))
end

@testset "Profit gain DDDC" begin
    α = Float64(0.125)
    β = Float64(4e-1)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = Int(1e6) # 1e8
    price_index = 1:n_prices

    competition_params_dict = Dict(
        :high => CompetitionParameters(0.25, -0.75, (2, 2), (0, 0)),
        :low => CompetitionParameters(0.25, 0.75, (2, 2), (0, 0)),
    )

    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

    data_demand_digital_params = DataDemandDigitalParams(
        weak_signal_quality_level = 0.99,
        strong_signal_quality_level = 0.995,
        signal_is_strong = [true, false],
        frequency_high_demand = 0.9,
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

    # TODO: Until state handling is fixed for multi-agent simultaneous environments, we can't test this
    # RLBase.test_interfaces!(env)
    # RLBase.test_runnable!(AIAPCEnv(hyperparameters))

    for demand in [:high, :low]
        @test profit_gain(
            π(
                env.p_monop_opt[demand],
                env.p_monop_opt[demand],
                env.competition_params_dict[demand],
            )[1],
            env,
        )[demand] == 1
        @test profit_gain(
            π(
                env.p_Bert_nash_equilibrium[demand],
                env.p_Bert_nash_equilibrium[demand],
                env.competition_params_dict[demand],
            )[1],
            env,
        )[demand] == 0
    end
end

@testset "Profit gain check AIAPC" begin
    competition_params_dict = Dict(
        :high => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
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
    env.memory[1] = CartesianIndex(Int64(1), Int64(1))
    exper = Experiment(env; debug = true)

    # Find the Nash equilibrium profit
    params = env.competition_params_dict[:high]
    p_Bert_nash_equilibrium = exper.env.p_Bert_nash_equilibrium
    π_min_price =
        π(minimum(exper.env.price_options), minimum(exper.env.price_options), params)[1]

    π_nash = π(p_Bert_nash_equilibrium, p_Bert_nash_equilibrium, params)[1]
    @test π_nash > π_min_price
    for i = 1:capacity(exper.hook[Player(1)].hooks[2].rewards)
        push!(exper.hook[Player(1)].hooks[2].rewards, π_nash)
        push!(exper.hook[Player(2)].hooks[2].rewards, 0)
    end

    ec_summary_ = economic_summary(exper)
    # TODO: Convergence profit needs to be tested with a properly configured tabular approximator...
    # @test round(profit_gain(ec_summary_.convergence_profit[1], env); digits = 2) == 0
    # @test round(profit_gain(ec_summary_.convergence_profit[2], env); digits = 2) == 1.07

    p_monop_opt = exper.env.p_monop_opt
    π_monop = π(p_monop_opt, p_monop_opt, params)[1]
    π_max_price =
        π(maximum(exper.env.price_options), maximum(exper.env.price_options), params)[1]
    @test π_max_price < π_monop

    for i = 1:capacity(exper.hook[Player(1)].hooks[2].rewards)
        push!(exper.hook[Player(1)].hooks[2].rewards, π_monop)
        push!(exper.hook[Player(2)].hooks[2].rewards, 0)
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
        :high => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
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
    @test action_space(env, Player(1)) == Int64.(1:15)
    @test reward(env) != 0 # reward reflects outcomes of last play (which happens at player = 1, e.g. before any actions chosen)
    act!(env, CartesianIndex(Int64(5), Int64(5)))
    @test reward(env) != [0, 0] # reward is zero as at least one player has already played (technically sequental plays)
end

@testset "run AIAPC multiprocessing code" begin
    n_procs_ = 1

    _procs = addprocs(
        Sys.CPU_THREADS,
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
        :high => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
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

@testset "run full DDDC simulation low-low" begin
    α = Float64(0.125)
    β = Float64(4e-1)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = Int(1e6)
    price_index = 1:n_prices

    competition_params_dict = Dict(
        :low => CompetitionParameters(0.25, 0.75, (2, 2), (0, 0)),
        :high => CompetitionParameters(0.25, -0.75, (2, 2), (0, 0)),
    )

    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

    data_demand_digital_params = DataDemandDigitalParams(
        weak_signal_quality_level = 1,
        strong_signal_quality_level = 1,
        signal_is_strong = [false, false],
        frequency_high_demand = 0.7,
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

    e_out = run(hyperparams; stop_on_convergence = true)
    e_sum = economic_summary(e_out)
    @test e_out.hook[Player(1)][2].demand_state_high_vect[end] ==
          (e_out.env.memory.demand_state == :high)

    player_ = Player(1)
    demand_state_high_vect = [e_out.hook[player_][2].demand_state_high_vect...]
    rewards = [e_out.hook[player_][2].rewards...]
    @test mean(rewards[demand_state_high_vect]) ≈ e_sum.convergence_profit_demand_high[1] atol =
        1e-2
    @test mean(rewards[.!demand_state_high_vect]) ≈ e_sum.convergence_profit_demand_low[1] atol =
        1e-2
    @test mean(e_out.env.profit_array[:, :, :, 1]) >
          mean(e_out.env.profit_array[:, :, :, 2])
    @test 0.65 < e_sum.percent_demand_high < 0.75
    @test all(e_sum.convergence_profit_demand_high > e_sum.convergence_profit_demand_low)
    @test all(1 .> e_sum.profit_gain .> 0)
    @test all(1 .> e_sum.profit_gain_demand_low .> 0)
    @test all(1 .> e_sum.profit_gain_demand_high .> 0)
    @test extract_profit_vars(e_out.env) == (
        Dict(:high => 0.2386460385715974, :low => 0.19331233681405383),
        Dict(:high => 0.4317126027908472, :low => 0.25),
    )

    @test extract_profit_vars(e_out.env) == (
        Dict(:high => 0.2386460385715974, :low => 0.19331233681405383),
        Dict(:high => 0.4317126027908472, :low => 0.25),
    )

    @test extract_quantity_vars(e_out.env)[1][:high] >
          extract_quantity_vars(e_out.env)[1][:low]
    @test extract_quantity_vars(e_out.env)[2][:high] >
          extract_quantity_vars(e_out.env)[2][:low]
    @test all(e_sum.price_response_to_demand_signal_mse .> 0)
end

@testset "run full DDDC simulation high-low" begin
    α = Float64(0.125)
    β = Float64(4e-1)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = Int(1e6)
    price_index = 1:n_prices

    competition_params_dict = Dict(
        :low => CompetitionParameters(0.25, 0.75, (2, 2), (0, 0)),
        :high => CompetitionParameters(0.25, -0.75, (2, 2), (0, 0)),
    )

    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

    data_demand_digital_params = DataDemandDigitalParams(
        weak_signal_quality_level = 1,
        strong_signal_quality_level = 1,
        signal_is_strong = [true, false],
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

    e_out = run(hyperparams; stop_on_convergence = true)
    e_sum = economic_summary(e_out)

    for player_ in [1, 2]
        @test e_out.hook[Player(player_)][2].demand_state_high_vect[end] ==
              (e_out.env.memory.demand_state == :high)
        demand_state_high_vect = [e_out.hook[Player(player_)][2].demand_state_high_vect...]
        rewards = [e_out.hook[Player(player_)][2].rewards...]
        @test mean(rewards[demand_state_high_vect]) ≈
              e_sum.convergence_profit_demand_high[player_] atol = 1e-2
        @test mean(rewards[.!demand_state_high_vect]) ≈
              e_sum.convergence_profit_demand_low[player_] atol = 1e-2
        @test mean(e_out.hook[Player(player_)][1].best_response_vector .== 0) < 0.05
    end

    @test mean(e_out.env.profit_array[:, :, :, 1]) >
          mean(e_out.env.profit_array[:, :, :, 2])
    @test 0.45 < e_sum.percent_demand_high < 0.55

    @test all(e_sum.convergence_profit_demand_high > e_sum.convergence_profit_demand_low)
    @test any(1 .> e_sum.profit_gain .> 0)
    @test any(1 .> e_sum.profit_gain_demand_low .> 0)
    @test extract_profit_vars(e_out.env) == (
        Dict(:high => 0.2386460385715974, :low => 0.19331233681405383),
        Dict(:high => 0.4317126027908472, :low => 0.25),
    )

    @test extract_profit_vars(e_out.env) == (
        Dict(:high => 0.2386460385715974, :low => 0.19331233681405383),
        Dict(:high => 0.4317126027908472, :low => 0.25),
    )

    @test extract_quantity_vars(e_out.env)[1][:high] >
          extract_quantity_vars(e_out.env)[1][:low]
    @test extract_quantity_vars(e_out.env)[2][:high] >
          extract_quantity_vars(e_out.env)[2][:low]
    @test all(e_sum.price_response_to_demand_signal_mse .> 0)
end

@testset "run full DDDC simulation high-high" begin
    α = Float64(0.125)
    β = Float64(4e-1)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = Int(1e6)
    price_index = 1:n_prices

    competition_params_dict = Dict(
        :low => CompetitionParameters(0.25, 0.75, (2, 2), (0, 0)),
        :high => CompetitionParameters(0.25, -0.75, (2, 2), (0, 0)),
    )

    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

    data_demand_digital_params = DataDemandDigitalParams(
        weak_signal_quality_level = 1,
        strong_signal_quality_level = 1,
        signal_is_strong = [true, true],
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

    e_out = run(hyperparams; stop_on_convergence = true)
    e_sum = economic_summary(e_out)
    @test e_out.hook[Player(1)][2].demand_state_high_vect[end] ==
          (e_out.env.memory.demand_state == :high)
    demand_state_high_vect = [e_out.hook[Player(1)][2].demand_state_high_vect...]
    rewards = [e_out.hook[Player(1)][2].rewards...]
    @test mean(rewards[demand_state_high_vect]) ≈ e_sum.convergence_profit_demand_high[1] atol =
        1e-2
    @test mean(rewards[.!demand_state_high_vect]) ≈ e_sum.convergence_profit_demand_low[1] atol =
        1e-2
    @test mean(e_out.env.profit_array[:, :, :, 1]) >
          mean(e_out.env.profit_array[:, :, :, 2])
    @test 0.45 < e_sum.percent_demand_high < 0.55
    @test all(e_sum.convergence_profit_demand_high > e_sum.convergence_profit_demand_low)
    @test all(1 .> e_sum.profit_gain .> 0)
    @test all(1 .> e_sum.profit_gain_demand_high .> 0)
    @test_broken extract_profit_vars(e_out.env) == (
        Dict(:high => 0.2386460385715974, :low => 0.19331233681405383),
        Dict(:high => 0.4317126027908472, :low => 0.25),
    )

    @test_broken extract_profit_vars(e_out.env) == (
        Dict(:high => 0.2386460385715974, :low => 0.19331233681405383),
        Dict(:high => 0.4317126027908472, :low => 0.25),
    )

    @test extract_quantity_vars(e_out.env)[1][:high] >
          extract_quantity_vars(e_out.env)[1][:low]
    @test extract_quantity_vars(e_out.env)[2][:high] >
          extract_quantity_vars(e_out.env)[2][:low]
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
        :high => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
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
    @test minimum(c_out.policy[Player(1)].policy.learner.approximator.model) < 6
    @test maximum(c_out.policy[Player(1)].policy.learner.approximator.model) > 5.5

    # ensure that the policy is updated by the learner
    @test sum(c_out.policy[Player(1)].policy.learner.approximator.model; dims = 2) != 0
    state_sum = sum(c_out.policy[Player(1)].policy.learner.approximator.model; dims = 1)
    @test !all(y -> y == state_sum[1], state_sum)
    @test length(reward(c_out.env)) == 2
    @test length(reward(c_out.env, 1)) == 1

    c_out.env.is_done[1] = false
    @test reward(c_out.env) == (0, 0)
    @test reward(c_out.env, 1) != 0

    @test sum(c_out.hook[Player(1)][1].best_response_vector == 0) == 0
    @test c_out.hook[Player(1)][1].best_response_vector !=
          c_out.hook[Player(2)][1].best_response_vector
end

@testset "Run a set of AIAPC experiments." begin
    competition_params_dict = Dict(
        :high => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
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
            convergence_threshold = 1000,
        ) for α in α_ for β in β_
    ]

    experiments = @chain hyperparameter_vect run_and_extract.(stop_on_convergence = true)

    @test experiments[1] isa AIAPCSummary
    @test all(10 < experiments[1].iterations_until_convergence[i] < max_iter for i = 1:2)
    @test (
        sum(experiments[1].convergence_profit .> 1) +
        sum(experiments[1].convergence_profit .< 0)
    ) == 0
    @test experiments[1].convergence_profit[1] != experiments[1].convergence_profit[2]
    @test all(experiments[1].is_converged)
end

@testset "AIAPC Hyperparameter Set" begin
    α_vect = Float64.(range(0.0025, 0.25, 3))
    β_vect = Float64.(range(0.025, 2, 3))
    max_iter = Int(1e8)
    convergence_threshold = 10
    δ = 0.95
    n_parameter_iterations = 2

    competition_params_dict = Dict(
        :high => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
    )
    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

    hyperparameter_vect = build_hyperparameter_set(
        α_vect,
        β_vect,
        δ,
        max_iter,
        competition_solution_dict,
        convergence_threshold,
        n_parameter_iterations,
    )

    df_1 = DataFrame([
        (hyperparameter.α, hyperparameter.β, 1) for hyperparameter in hyperparameter_vect
    ])
    rename!(df_1, [:α, :β, :count])
    df_ = @chain df_1 @groupby(:α, :β) @combine(length(:count))
    @test all(df_[!, :count_length] .== 2)
end

@testset "Parameter / learning checks" begin
    α = Float64(0.125)
    β = Float64(1)
    δ = 0.95
    ξ = 0.1
    δ = 0.95
    n_prices = 15
    max_iter = Int64(5e6)
    price_index = 1:n_prices

    competition_params_dict = Dict(
        :high => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
    )
    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

    hyperparameters = AIAPCHyperParameters(
        α,
        β,
        δ,
        max_iter,
        competition_solution_dict;
        convergence_threshold = 500,
    )


    c_out = run(hyperparameters; stop_on_convergence = true, debug = true)

    # ensure that the policy is updated by the learner
    @test sum(c_out.policy[Player(1)].policy.learner.approximator.model .!= 0) != 0
    @test sum(c_out.policy[Player(2)].policy.learner.approximator.model .!= 0) != 0
    @test c_out.env.is_done[1]
    @test 500 < c_out.hook[Player(1)][1].iterations_until_convergence <= max_iter
    @test 500 < c_out.hook[Player(2)][1].iterations_until_convergence <= max_iter


    @test c_out.policy[Player(1)].trajectory.container[:reward][1] .!= 0
    @test c_out.policy[Player(2)].trajectory.container[:reward][1] .!= 0

    @test c_out.policy[Player(1)].policy.learner.approximator.model !=
          c_out.policy[Player(2)].policy.learner.approximator.model
    @test c_out.hook[Player(1)][1].best_response_vector !=
          c_out.hook[Player(2)][1].best_response_vector


    @test mean(
        argmax(c_out.policy[Player(1)].policy.learner.approximator.model, dims = 1) .!=
        argmax(c_out.policy[Player(2)].policy.learner.approximator.model, dims = 1),
    ) < 0.98

    for i in [Player(1), Player(2)]
        @test c_out.hook[i][1].convergence_duration >= 0
        @test c_out.hook[i][1].is_converged
        @test c_out.hook[i][1].convergence_threshold == 500
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
        :high => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
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
    @test RLFarm.get_ϵ(c_out.policy[Player(1)].policy.explorer) < 1e-4
    @test RLFarm.get_ϵ(c_out.policy[Player(2)].policy.explorer) < 1e-4
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

    policy = RandomPolicy()
    competition_params_dict = Dict(
        :high => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
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
    @test 0.98 < RLFarm.get_ϵ(c_out.policy[Player(1)].policy.explorer) < 1
    @test 0.98 < RLFarm.get_ϵ(c_out.policy[Player(2)].policy.explorer) < 1

    @test RLCore.check!(c_out.stop_condition, policy, c_out.env) == true
    @test RLCore.check!(c_out.stop_condition.stop_conditions[1], policy, c_out.env) == false
    @test RLCore.check!(c_out.stop_condition.stop_conditions[2], policy, c_out.env) == true

    @test c_out.hook[Player(1)][1].convergence_duration >= 5
    @test c_out.hook[Player(2)][1].convergence_duration >= 5
    @test (c_out.hook[Player(2)][1].convergence_duration == 5) ||
          (c_out.hook[Player(1)][1].convergence_duration == 5)
end

@testset "run DDDC multiprocessing code" begin
    _procs = addprocs(
        Sys.CPU_THREADS,
        topology = :master_worker,
        exeflags = ["--threads=1", "--project=$(Base.active_project())"],
    )

    @everywhere begin
        using Pkg
        Pkg.instantiate()
        using AlgorithmicCompetition
    end

    for debug in [true, false]
        AlgorithmicCompetition.run_dddc(
            n_parameter_iterations = 1,
            max_iter = Int(1e4),
            convergence_threshold = Int(1e2),
            n_grid_increments = 2,
            debug = debug,
        )
    end
    rmprocs(_procs)
end
