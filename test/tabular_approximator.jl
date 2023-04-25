using Flux
using AlgorithmicCompetition: TabularApproximator, TabularVApproximator, TabularQApproximator, TDLearner
using Test

@testset "TabularApproximator" begin
    table_q = zeros(Float32, 10, 200)
    table_v = zeros(Float32, 200)
    optimizer_ = Descent(0.125)

    tabular_q_approx = TabularApproximator(table_q, Descent(0.125))
    @test tabular_q_approx.table == table_q
    @test tabular_q_approx.optimizer isa Descent
    @test tabular_q_approx(10) isa SubArray
    @test tabular_q_approx(10, 1) isa Float32

    tabular_v_approx = TabularApproximator(table_v, Descent(0.125))
    @test tabular_v_approx.table == table_v
    @test tabular_v_approx.optimizer isa Descent
    @test tabular_v_approx(10) isa Float32

    @test TabularVApproximator(; n_state=10).table == zeros(Float32, 10)
    @test TabularQApproximator(; n_state=10, n_action=2).table == zeros(Float32, 2, 10)
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
