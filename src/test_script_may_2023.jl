using Test
using JuMP
using Chain
using ReinforcementLearningCore:
    PostActStage,
    PreActStage,
    state,
    reward,
    current_player,
    action_space,
    EpsilonGreedyExplorer,
    RandomPolicy,
    MultiAgentPolicy,
    RLCore,
    ResetAtTerminal
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
    Q,
    run,
    run_and_extract,
    Experiment,
    reward,
    InitMatrix,
    get_ϵ,
    AIAPCEpsilonGreedyExplorer,
    AIAPCSummary,
    TDLearner,
    TabularApproximator
using JET
using ProfileView
# using Distributed

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

hyperparams =
    AIAPCHyperParameters(α, β, δ, max_iter, competition_solution; convergence_threshold = 1)

env = AIAPCEnv(hyperparams)
experiment = Experiment(env; stop_on_convergence = false)

@report_opt Base.push!(experiment.policy, PreActStage(), experiment.env)
@report_opt RLBase.plan!(experiment.policy, experiment.env)

@time run(hyperparams; stop_on_convergence = false);
@time run(hyperparams; stop_on_convergence = false);
# @profview run(hyperparams; stop_on_convergence = false);
@report_opt RLCore._run(
    experiment.policy,
    experiment.env,
    experiment.stop_condition,
    experiment.hook,
    ResetAtTerminal(),
)

RLCore._run(
    experiment.policy,
    experiment.env,
    experiment.stop_condition,
    experiment.hook,
    ResetAtTerminal(),
)

# @profview RLCore._run(
#     experiment.policy,
#     experiment.env,
#     experiment.stop_condition,
#     experiment.hook,
#     ResetAtTerminal(),
# )

# RLBase.plan!(experiment.policy.agents[Symbol(1)], env)
# @report_opt RLBase.plan!(experiment.policy.agents[Symbol(1)], env, Symbol(1))
# @report_opt RLBase.plan!(experiment.policy, env, Symbol(1))
# @report_opt RLCore.forward(td_learner, env)
@report_opt RLBase.plan!(experiment.policy, env)
# @report_opt RLBase.act!(env, (1,1))


@report_opt AlgorithmicCompetition.economic_summary(experiment)
# n_procs_ = 3

# _procs = addprocs(
#     n_procs_,
#     topology = :master_worker,
#     exeflags = ["--threads=1", "--project=$(Base.active_project())"],
# )

# @everywhere begin
#     using Pkg
#     Pkg.instantiate()
#     using AlgorithmicCompetition
# end

# AlgorithmicCompetition.run_aiapc(;
#     n_parameter_iterations = 100,
#     # max_iter = Int(1e6),
#     # convergence_threshold = Int(10),
# )

# rmprocs(_procs)
