using Test
using JuMP
using Chain
using DataFrames
using DataFrameMacros
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
using Statistics
using AlgorithmicCompetition:
    AIAPCEnv,
    AIAPCEpsilonGreedyExplorer,
    AIAPCHyperParameters,
    AIAPCPolicy,
    AIAPCSummary,
    AlgorithmicCompetition,
    build_hyperparameter_set,
    CompetitionParameters,
    CompetitionParameters,
    CompetitionSolution,
    construct_AIAPC_action_space,
    construct_AIAPC_profit_array,
    construct_AIAPC_state_space_lookup,
    construct_DDDC_action_space,
    construct_DDDC_profit_array,
    construct_DDDC_state_space_lookup,
    ConvergenceCheck,
    DataDemandDigitalParams,
    DDDCEnv,
    DDDCHyperParameters,
    economic_summary,
    Experiment,
    extract_profit_vars,
    extract_quantity_vars,
    get_demand_level,
    get_demand_signals,
    get_ϵ,
    initialize_price_memory,
    InitMatrix,
    p_BR,
    post_prob_high_low_given_signal,
    profit_gain,
    Q_i_0,
    Q,
    Q!,
    reward,
    run_and_extract,
    run,
    solve_bertrand,
    solve_monopolist,
    TDLearner,
    π
using Distributed

include("alpha_beta.jl")
include("stochastic_demand_stochastic_information.jl")
include("competitive_equilibrium.jl")
include("hooks.jl")
include("explorer.jl")

include("tabular_approximator.jl")
include("q_learning.jl")
include("policy.jl")
include("integration.jl")
include("aiapc_conversion_check.jl")
