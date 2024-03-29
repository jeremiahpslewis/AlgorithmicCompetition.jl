@testset "get_demand_signals" begin
    @test get_demand_signals(true, [true, false], 0.0, 1.0) == [1, 0]
    @test get_demand_signals(true, [true, false], 1.0, 0.0) == [0, 1]
    @test get_demand_signals(false, [true, false], 0.0, 1.0) == [0, 1]
    @test get_demand_signals(false, [true, false], 1.0, 0.0) == [1, 0]
    @test all(
        4500 .<
        sum(get_demand_signals(false, [true, false], 0.5, 0.5) for i = 1:10000) .<
        5500,
    )
    @test get_demand_signals(false, [true, false], 1.0, 1.0) == [0, 0]
    @test get_demand_signals(false, [true, true], 0.5, 1.0) == [0, 0]
    @test get_demand_signals(true, [true, true], 0.5, 1.0) == [1, 1]
end

@testset "get_demand_level" begin
    @test get_demand_level(1.0) == true
    @test get_demand_level(0.0) == false
end

@testset "construct_action_space" begin
    @test length(construct_AIAPC_action_space(1:15)) == 225
    @test length(construct_DDDC_action_space(1:15)) == 900
end

@testset "initialize_price_memory" begin
    @test length(initialize_price_memory(1:15, 2)) == 2
end


@testset "post_prob_high_low_given_signal" begin
    @test post_prob_high_low_given_signal(0, 1)[2] == 1
    @test post_prob_high_low_given_signal(1, 0)[2] == 0.0
    @test post_prob_high_low_given_signal(0.5, 0.5) == [0.5, 0.5]
end
