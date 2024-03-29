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
