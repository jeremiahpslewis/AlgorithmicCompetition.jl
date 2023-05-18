using AlgorithmicCompetition:
    AlgorithmicCompetition,
    CompetitionParameters,
    CompetitionSolution,
    AIAPCHyperParameters,
    AIAPCEnv,
    Experiment,
    run_and_extract,
    run
using BenchmarkTools

competition_params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))

competition_solution = CompetitionSolution(competition_params)

n_increments = 100
max_iter = Int(1e6) # Should be 1e9
α_ = convert.(Float32, range(0.025, 0.25, n_increments))
β_ = convert.(Float32, range(1.25e-8, 2e-5, n_increments))
δ = 0.95

hyperparameter_vect =
    [AIAPCHyperParameters(α, β, δ, max_iter, competition_solution) for α in α_ for β in β_]

@btime run(hyperparameter_vect[1]; stop_on_convergence = false)

# a = run(hyperparameter_vect[1]; stop_on_convergence=false)

# using ReinforcementLearning
# @report_opt RLBase.reward(a.env)
# a = run(hyperparameter_vect[1]; stop_on_convergence=false)

# a


# Trying half-precision
# Try cutting out action space calls



# function reward1(env::AIAPCEnv)
#     @views env.is_done[1] ? (env.profit_array[env.memory, :]) : [Float32(0), Float32(0)]
# end

# @btime reward1(a.env)
# 322.022 ns (9 allocations: 464 bytes)
# 113.926 ns (2 allocations: 288 bytes)
# 111.879 ns (2 allocations: 288 bytes)

# function reward1(env::AIAPCEnv, p::Int)
#     env.is_done[1] ? env.profit_array[env.memory[1], env.memory[2], p] : 0
# end

# @btime reward1(a.env)
