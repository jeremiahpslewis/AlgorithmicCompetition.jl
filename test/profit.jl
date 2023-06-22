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
    profit_array = construct_profit_array(price_options, competition_solution.params, 2)

    profit_array[5, 3, :] ≈
    π(price_options[5], price_options[3], competition_solution.params)
end
