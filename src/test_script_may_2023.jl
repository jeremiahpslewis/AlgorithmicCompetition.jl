using Test
using JuMP
using JET
using Chain
using BenchmarkTools
using ReinforcementLearningCore:
    PostActStage,
    PreActStage,
    PreEpisodeStage,
    PostEpisodeStage,
    state,
    reward,
    current_player,
    action_space,
    EpsilonGreedyExplorer,
    RandomPolicy,
    MultiAgentPolicy,
    RLCore,
    ResetAtTerminal
using ReinforcementLearningBase:
    RLBase, test_interfaces!, test_runnable!, AbstractPolicy, optimise!, act!
import ReinforcementLearningCore
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
    extract_sim_results
# using JET
# using ProfileView
using Distributed

# RLCore.TimerOutputs.enable_debug_timings(RLCore)

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


hyperparams = AIAPCHyperParameters(
    α,
    β,
    δ,
    max_iter,
    competition_solution_dict;
    convergence_threshold = Int(1e5),
)

env = AIAPCEnv(hyperparams)
experiment = Experiment(env; stop_on_convergence = false)

push!(experiment.policy, PreEpisodeStage(), experiment.env)
push!(experiment.policy, PreActStage(), experiment.env)
optimise!(experiment.policy, PreActStage())
actions = RLBase.plan!(experiment.policy, experiment.env)
act!(experiment.env, actions)
push!(experiment.policy, PostActStage(), experiment.env, actions)
optimise!(experiment.policy, PostActStage())
push!(experiment.policy, PostEpisodeStage(), env)

@report_opt Base.push!(
    experiment.policy,
    PostActStage(),
    experiment.env,
    CartesianIndex(1, 1),
)
@report_opt RLBase.plan!(experiment.policy, experiment.env)
@report_opt optimise!(experiment.policy, PostActStage())

@time run(hyperparams; stop_on_convergence = true);

RLCore.timer

a = @time run(hyperparams; stop_on_convergence = false);

# player = Symbol(1)
# next_state = state(env,player)
# action = 1

# @report_opt Base.push!(experiment.policy[player].trajectory, (state = next_state, action = action, reward = reward(env, player), terminal = true))

# experiment.policy[player].trajectory.container.traces

# using ReinforcementLearningTrajectories: Traces

# @generated function push12(ts, xs::NamedTuple{N,T}) where {N,T}
#     ex = :()
#     for n in N
#         ex = :($ex; push!(ts, Val($(QuoteNode(n))), xs.$n))
#     end
#     return :($ex)
# end


# push12(Int64[1,2,3], (a = 1, b = 2, c = 3))

# push7(experiment.policy[Symbol(1)].trajectory.container.traces, (state = next_state, action = action, reward = reward(env, player), terminal = true))
