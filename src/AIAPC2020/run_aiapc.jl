import ProgressMeter: @showprogress
using Distributed
using CSV

function run_aiapc(; n_parameter_iterations=1, csv_out_path="", max_iter=Int(1e9))
    competition_params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])
    competition_solution = CompetitionSolution(competition_params)
    n_increments = 100
    max_iter = Int(1e9)
    α_ = Float32.(range(0.025, 0.25, n_increments))
    β_ = Float32.(range(1.25e-8, 2e-5, n_increments))
    δ = 0.95

    hyperparameter_vect = [
        AIAPCHyperParameters(α, β, δ, max_iter, competition_solution) for α in α_ for β in β_
    ]

    exp_list_ = AIAPCSummary[]
    for i in 1:n_parameter_iterations
        println("Running iteration $i of $n_parameter_iterations")
        exp_list = @showprogress pmap(run_and_extract, sample(hyperparameter_vect, 1000); on_error=identity)
        append!(exp_list_, exp_list)
        CSV.write(csv_out_path * "simulation_results_$i.csv", df)
    end

    return exp_list_
end
