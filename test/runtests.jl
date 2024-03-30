using JuMP
using Chain
using DataFrames
using DataFrameMacros
using Test
using ReinforcementLearning:
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
    optimise!,
    RLBase, AbstractPolicy, act!, plan!
import ReinforcementLearning: RLCore
using Statistics
using AlgorithmicCompetition:
    AIAPCEnv,
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
    DDDCTotalRewardPerLastNEpisodes,
    DDDCPolicy,
    economic_summary,
    Experiment,
    extract_profit_vars,
    extract_quantity_vars,
    get_demand_level,
    get_demand_signals,
    initialize_price_memory,
    InitMatrix,
    p_BR,
    post_prob_high_low_given_signal,
    profit_gain,
    Q_i_0,
    Q,
    reward,
    run_and_extract,
    run,
    solve_bertrand,
    solve_monopolist,
    TDLearner,
    π
using Distributed

@testset "AlgorithmicCompetition.jl" begin
    @testset "Paramter tests" begin
        include("alpha_beta.jl")
        include("stochastic_demand_stochastic_information.jl")
        include("competitive_equilibrium.jl")
    end
    @testset "RL.jl structs" begin
        include("hooks.jl")
        include("explorer.jl")
        include("tabular_approximator.jl")
        include("q_learning.jl")
        include("policy.jl")
    end
    @testset verbose = true "Integration tests" begin
        include("integration.jl")
    end
    @testset "Output tests" begin
        include("aiapc_conversion_check.jl")
    end
end
