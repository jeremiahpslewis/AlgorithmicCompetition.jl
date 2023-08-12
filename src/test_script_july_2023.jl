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
using Statistics
using AlgorithmicCompetition:
    AlgorithmicCompetition,
    CompetitionParameters,
    DDDCHyperParameters,
    CompetitionParameters,
    AIAPCHyperParameters,
    AIAPCPolicy,
    AIAPCEnv,
    CompetitionSolution,
    ConvergenceCheck,
    DataDemandDigitalParams,
    solve_monopolist,
    solve_bertrand,
    p_BR,
    construct_AIAPC_state_space_lookup,
    construct_AIAPC_profit_array,
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
    TabularApproximator,
    economic_summary,
    extract_sim_results,
    DDDCEnv
using JET
using BenchmarkTools
# using ProfileView
using Distributed

@time exp_list = AlgorithmicCompetition.run_dddc(;
    n_parameter_iterations = 10,
    max_iter = Int(1e9),
    n_grid_increments = 25,
)

# RLCore.TimerOutputs.enable_debug_timings(RLCore)

α = Float64(0.125)
β = Float64(4e-1)
δ = 0.95
ξ = 0.1
δ = 0.95
n_prices = 15
max_iter = Int(1e6) # 1e8
price_index = 1:n_prices

competition_params_dict = Dict(
    :high => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
    :low => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
)

competition_solution_dict =
    Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

# NOTE: low quality probability 0.5+x, high quality boost, is high quality signal, high demand freq
data_demand_digital_params = DataDemandDigitalParams(
    low_signal_quality_level = 0.99,
    high_signal_quality_level = 0.995,
    signal_quality_is_high = [true, false],
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
experiment = Experiment(env; stop_on_convergence = true)
ex = @time run(hyperparams; stop_on_convergence = true);
economic_summary(ex)
extract_sim_results([economic_summary(ex)])
# @report_opt Base.push!(
#     experiment.policy,
#     PostActStage(),
#     experiment.env,
#     CartesianIndex(1, 1),
# )
# @report_opt RLBase.plan!(experiment.policy, experiment.env)

@report_opt RLCore._run(
    experiment.policy,
    experiment.env,
    experiment.stop_condition,
    experiment.hook,
    ResetAtTerminal(),
)

ex = @time run(hyperparams; stop_on_convergence = true);
economic_summary(ex)
# a = @time run(hyperparams; stop_on_convergence = false);

# ex.hook[Symbol(1)][2].convergence_duration
# RLCore.to

# a = @time run(hyperparams; stop_on_convergence = false);


# # TODO Debug how this can happen...
# # a.hook[Symbol(1)][1].rewards[end-10:end] Weird oscillation happening here...only for player 1




# # AlgorithmicCompetition.π(a.env.price_options[7], a.env.price_options[7], a.env.competition_params)


# # next_price_set = get_optimal_action(a.env, last_observed_state)
# # next_state = get_state_from_memory(a.env, next_price_set)

# # for i in 1:20
# #     a = run(hyperparams; stop_on_convergence = true)
# # end

# a_list = run_and_extract.(repeat([hyperparams], 10))
# mean(profit_gain.([g.convergence_profit for g in a_list], (a.env,)))

# # @report_opt push!(hook, PostEpisodeStage(), 1.0, 1.0)
# # a.policy.agents[Symbol(1)].trajectory.container[:next_state]
# # 509.952693 seconds (8.40 G allocations: 330.871 GiB, 8.58% gc time, 0.10% compilation time)
# # 626.709955 seconds (8.60 G allocations: 333.888 GiB, 7.34% gc time, 0.10% compilation time) # with circulararraybuffers


# # 53.278634 seconds (859.40 M allocations: 33.351 GiB, 8.71% gc time) # With circular arrays
# # 50.567420 seconds (839.40 M allocations: 33.055 GiB, 8.84% gc time) # With normal hook

# # @profview run(hyperparams; stop_on_convergence = false);
# @report_opt RLCore._run(
#     experiment.policy,
#     experiment.env,
#     experiment.stop_condition,
#     experiment.hook,
#     ResetAtTerminal(),
# )

# RLCore._run(
#     experiment.policy,
#     experiment.env,
#     experiment.stop_condition,
#     experiment.hook,
#     ResetAtTerminal(),
# )

# # @profview RLCore._run(
# #     experiment.policy,
# #     experiment.env,
# #     experiment.stop_condition,
# #     experiment.hook,
# #     ResetAtTerminal(),
# # )

# # RLBase.plan!(experiment.policy.agents[Symbol(1)], env)
# # @report_opt RLBase.plan!(experiment.policy.agents[Symbol(1)], env, Symbol(1))
# # @report_opt RLBase.plan!(experiment.policy, env, Symbol(1))
# # @report_opt RLCore.forward(td_learner, env)
# @report_opt RLBase.plan!(experiment.policy, env)
# # @report_opt RLBase.act!(env, (1,1))

# # AlgorithmicCompetition.run_and_extract(hyperparams; stop_on_convergence = true).iterations_until_convergence[1]
# @report_opt AlgorithmicCompetition.economic_summary(experiment)
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
