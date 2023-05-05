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
    PreActStage
using ReinforcementLearningBase: test_interfaces!, test_runnable!, AbstractPolicy
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
