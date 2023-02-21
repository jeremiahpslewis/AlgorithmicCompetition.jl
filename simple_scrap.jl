using AlgorithmicCompetition:
AlgorithmicCompetition,
CompetitionParameters,
CompetitionSolution,
CalvanoHyperParameters,
CalvanoEnv,
Experiment,
run_and_extract,
run
using BenchmarkTools

competition_params = CompetitionParameters(0.25, 0, [2, 2], [1, 1])

competition_solution = CompetitionSolution(competition_params)

n_increments = 100
max_iter = Int(1e6) # Should be 1e9
α_ = range(0.025, 0.25, n_increments)
β_ = range(1.25e-8, 2e-5, n_increments)
δ = 0.95

hyperparameter_vect = [
CalvanoHyperParameters(α, β, δ, max_iter, competition_solution) for α in α_ for β in β_
]

@btime run_and_extract(hyperparameter_vect[1]; stop_on_convergence=false)

# a = run(hyperparameter_vect[1]; stop_on_convergence=false)

# a
# Try int8 action space
# TODO: try @views instead of @view
