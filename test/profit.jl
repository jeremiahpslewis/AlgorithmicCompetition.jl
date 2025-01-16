@testset "Profit array test" begin
    competition_params_dict = Dict(
        :high => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (0, 0)),
    )
    competition_solution_dict =
        Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])
    params = AIAPCHyperParameters(
        Float64(0.1),
        Float64(1e-4),
        0.95,
        Int(1e7),
        competition_solution_dict,
    )
    env = params |> AIAPCEnv
    exper = Experiment(env)

    price_options = env.price_options
    action_space_ = env.action_space
    profit_array = construct_profit_array(
        price_options,
        competition_solution_dict[:high].params,
        2;
        false,
        :high,
    )

    profit_array[5, 3, :] ≈
    π(price_options[5], price_options[3], competition_solution_dict[:high].params)
end
