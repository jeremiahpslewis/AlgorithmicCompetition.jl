using Test
using Flux
using AlgorithmicCompetition:
    TabularApproximator, TabularVApproximator, TabularQApproximator, TDLearner, QBasedPolicy
import ReinforcementLearningBase: RLBase
using ReinforcementLearningCore
using ReinforcementLearningEnvironments


@testset "Constructors" begin
    @test TabularApproximator(fill(1, 10, 10), fill(1, 10)) isa TabularApproximator
    @test TabularVApproximator(n_state = 10) isa
          TabularApproximator{1,Vector{Float64},InvDecay}
    @test TabularQApproximator(n_state = 10, n_action = 10) isa
          TabularApproximator{2,Matrix{Float64},InvDecay}
end


@testset "RLCore.forward" begin
    v_approx = TabularVApproximator(n_state = 10)
    @test RLCore.forward(v_approx, 1) == 0.0

    q_approx = TabularQApproximator(n_state = 5, n_action = 10)
    @test RLCore.forward(q_approx, 1) == zeros(Float64, 10)
    @test RLCore.forward(q_approx, 1, 5) == 0.0

end


# using Flux
# td_learner = TDLearner(;
# # TabularQApproximator with specified init matrix
# approximator = TabularApproximator(
#     InitMatrix(experiment.env.n_prices, experiment.env.n_state_space),
#     Descent(experiment.env.α),
# ),
# # For param info: https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/f97747923c6d7bbc5576f81664ed7b05a2ab8f1e/src/ReinforcementLearningZoo/src/algorithms/tabular/td_learner.jl#L15
# method = :SARS,
# γ = experiment.env.δ,
# n = 0,
# )

# RLBase.optimise!(td_learner, experiment.policy.agents[Symbol(1)].trajectory.container)

# @testset "TabularApproximator" begin
#     n_state = 200
#     n_action = 10
#     optimizer_ = Descent(0.125)

#     table_q = zeros(Float64, n_action, n_state)
#     tabular_q_approx = TabularApproximator(table_q, Descent(0.125))
#     @test tabular_q_approx.table == table_q
#     @test tabular_q_approx.optimizer isa Descent
#     @test tabular_q_approx(10) isa SubArray
#     @test tabular_q_approx(10, 1) isa Float64
#     RLBase.optimise!(tabular_q_approx, (10, 1) => 1.0)
#     @test tabular_q_approx.table[1, 10] == -0.125f0
#     RLBase.optimise!(tabular_q_approx, 11 => fill(1.0, n_action))
#     @test tabular_q_approx.table[1, 11] == -0.125f0

#     table_v = zeros(Float64, n_state)
#     tabular_v_approx = TabularApproximator(table_v, Descent(0.125))
#     @test tabular_v_approx.table == table_v
#     @test tabular_v_approx.optimizer isa Descent
#     @test tabular_v_approx(10) isa Float64
#     RLBase.optimise!(tabular_v_approx, 10 => 1.0)
#     @test tabular_v_approx.table[10] == -0.125f0

#     @test TabularVApproximator(; n_state = n_state, init = Float64(0.0)).table ==
#           zeros(Float64, n_state)
#     @test TabularQApproximator(;
#         n_state = n_state,
#         n_action = n_action,
#         init = Float64(0.0),
#     ).table == zeros(Float64, n_action, n_state)
# end


# @testset "TDLearner" begin
#     env = TicTacToeEnv()
#     td_learner = TDLearner(;
#         # TabularQApproximator with specified init matrix
#         approximator = TabularApproximator(
#             zeros(Float64, length(RLBase.action_space(env)), length(state_space(env))),
#             Descent(0.125),
#         ),
#         method = :SARS,
#         γ = 0.95,
#         n = 0,
#     )
#     @test td_learner(env) == zeros(Float64, RLBase.action_space(env))
#     @test td_learner(10) == zeros(Float64, RLBase.action_space(env))
#     @test td_learner(10, 1) == Float64(0)

#     nt_ = (; state = [fill(2, 2)], action = [fill(2, 2)], reward = 0.31, terminal = true)

#     RLBase.optimise!(td_learner, nt_)
#     @test td_learner(2, 2) != 0

#     nt_1 = (; state = [fill(3, 3)], action = [fill(3, 3)], reward = 0.31, terminal = true)
#     RLBase.optimise!(td_learner, nt_1)
#     @test td_learner(3, 3) != 0

#     transition_ = (1, 1, 1.0, false, 2)
#     @test RLBase.priority(td_learner, transition_) != 0
# end

# @testset "QBasedPolicy" begin
#     env = TicTacToeEnv()

#     q_policy = QBasedPolicy(;
#         learner = TDLearner(;
#             # TabularQApproximator with specified init matrix
#             approximator = TabularApproximator(
#                 zeros(
#                     Float64,
#                     length(RLBase.action_space(env)),
#                     length(RLBase.state_space(env)),
#                 ),
#                 Descent(0.125),
#             ),
#             method = :SARS,
#             γ = 0.95,
#             n = 0,
#         ),
#         explorer = EpsilonGreedyExplorer(1, is_break_tie = true),
#     )
#     @test any([q_policy(env) != 1 for i = 1:10])

#     nt_ = (; state = [fill(2, 2)], action = [fill(2, 2)], reward = 0.31, terminal = true)
#     RLBase.optimise!(q_policy, nt_)
#     @test q_policy.learner(2, 2) != 0
# end

# @testset "MultiAgentQBasedPolicy" begin
#     env = RockPaperScissorsEnv()
#     multi_q_policy = MultiAgentPolicy((;
#         Symbol(1) => Agent(
#             QBasedPolicy(;
#                 learner = TDLearner(;
#                     # TabularQApproximator with specified init matrix
#                     approximator = TabularApproximator(
#                         zeros(
#                             Float64,
#                             length(RLBase.action_space(env)),
#                             length(RLBase.state_space(env)),
#                         ),
#                         Descent(0.125),
#                     ),
#                     method = :SARS,
#                     γ = 0.95,
#                     n = 0,
#                 ),
#                 explorer = EpsilonGreedyExplorer(1, is_break_tie = true),
#             ),
#             Trajectory(
#                 CircularArraySARSTraces(;
#                     capacity = 3,
#                     state = Int64 => (225,),
#                     action = Int64 => (15,),
#                     reward = Float64 => (),
#                     terminal = Bool => (),
#                 ),
#                 DummySampler(),
#             ),
#         ),
#         Symbol(2) => Agent(
#             QBasedPolicy(;
#                 learner = TDLearner(;
#                     # TabularQApproximator with specified init matrix
#                     approximator = TabularApproximator(
#                         zeros(
#                             Float64,
#                             length(RLBase.action_space(env)),
#                             length(RLBase.state_space(env)),
#                         ),
#                         Descent(0.125),
#                     ),
#                     method = :SARS,
#                     γ = 0.95,
#                     n = 0,
#                 ),
#                 explorer = EpsilonGreedyExplorer(1, is_break_tie = true),
#             ),
#             Trajectory(
#                 CircularArraySARSTraces(;
#                     capacity = 3,
#                     state = Int64 => (225,),
#                     action = Int64 => (15,),
#                     reward = Float64 => (),
#                     terminal = Bool => (),
#                 ),
#                 DummySampler(),
#             ),
#         ),
#     ))

#     multi_q_policy(PreActStage(), env)
#     env(multi_q_policy(env))
#     multi_q_policy(PostActStage(), env)

#     @test any([multi_q_policy[Symbol(1)](env) != 1 for i = 1:10])
#     @test any([multi_q_policy[Symbol(2)](env) != 1 for i = 1:10])
#     @test any([multi_q_policy[Symbol(1)](env) != 1 for i = 1:10])
#     @test any([multi_q_policy[Symbol(2)](env) != 1 for i = 1:10])
#     @test all([length([multi_q_policy(env)...]) for i = 1:10] .== 2)

#     for i = 1:2
#         RLBase.reset!(env)
#         multi_q_policy(PreActStage(), env)
#         env(multi_q_policy(env))
#         RLCore.optimise!(multi_q_policy)
#         multi_q_policy(PostActStage(), env)
#     end

#     @test any(multi_q_policy.agents[Symbol(1)].policy.learner.approximator.table .!= 0)
#     @test any(multi_q_policy.agents[Symbol(2)].policy.learner.approximator.table .!= 0)
# end
