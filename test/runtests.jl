using Test
using JuMP
using Chain
using ReinforcementLearningCore:
    RLCore,
    PostActStage,
    PreActStage,
    PostEpisodeStage,
    PreEpisodeStage,
    state,
    reward,
    current_player,
    action_space,
    EpsilonGreedyExplorer,
    RandomPolicy,
    MultiAgentPolicy,
    optimise!
using ReinforcementLearningBase:
    RLBase, test_interfaces!, test_runnable!, AbstractPolicy, act!, plan!
import ReinforcementLearningCore: RLCore
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
    construct_profit_array,
    Q,
    Q!,
    Q_i_0,
    π,
    run,
    run_and_extract,
    Experiment,
    reward,
    InitMatrix,
    get_ϵ,
    AIAPCEpsilonGreedyExplorer,
    AIAPCSummary,
    TDLearner,
    economic_summary,
    profit_gain,
    β_range,
    α_range,
    Int8
using Distributed

include("alpha_beta.jl")
include("competitive_equilibrium.jl")
include("hooks.jl")
include("policy.jl")
include("explorer.jl")

include("tabular_approximator.jl")
include("q_learning.jl")

include("integration.jl")
# include("aiapc_conversion_check.jl")
