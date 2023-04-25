using Flux
using AlgorithmicCompetition: TabularApproximator, TabularVApproximator, TabularQApproximator, TDLearner
using ReinforcementLearningBase
using Test

@testset "TabularApproximator" begin
    n_state = 200
    n_action = 10
    table_q = zeros(Float32, n_action, n_state)

    optimizer_ = Descent(0.125)

    tabular_q_approx = TabularApproximator(table_q, Descent(0.125))
    @test tabular_q_approx.table == table_q
    @test tabular_q_approx.optimizer isa Descent
    @test tabular_q_approx(10) isa SubArray
    @test tabular_q_approx(10, 1) isa Float32
    RLBase.optimise!(tabular_q_approx, (10,1) => 1.0)
    @test tabular_q_approx.table[1,10] == -0.125f0
    RLBase.optimise!(tabular_q_approx, 11 => fill(1.0, n_action))
    @test tabular_q_approx.table[1,11] == -0.125f0

    table_v = zeros(Float32, n_state)
    tabular_v_approx = TabularApproximator(table_v, Descent(0.125))
    @test tabular_v_approx.table == table_v
    @test tabular_v_approx.optimizer isa Descent
    @test tabular_v_approx(10) isa Float32
    RLBase.optimise!(tabular_v_approx, 10 => 1.0)
    @test tabular_v_approx.table[10] == -0.125f0

    @test TabularVApproximator(; n_state=n_state, init=Float32(0.0)).table == zeros(Float32, n_state)
    @test TabularQApproximator(; n_state=n_state, n_action=n_action, init=Float32(0.0)).table == zeros(Float32, n_action, n_state)
end



TDLearner(;
    # TabularQApproximator with specified init matrix
    approximator = TabularApproximator(
        zeros(Float32, 10, 200),
        Descent(0.125),
    ),
    method = :SARS,
    γ = 0.95,
    n = 0,
)
