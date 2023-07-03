@testset "get_demand_signals" begin
    @test get_demand_signals(true, [true, false], -0.5, 1.0) == [1, 0]
    @test get_demand_signals(true, [true, false], 0.5, -1.0) == [0, 1]
    @test get_demand_signals(false, [true, false], -0.5, 1.0) == [0, 1]
    @test get_demand_signals(false, [true, false], 0.5, -1.0) == [1, 0]
    @test all(4500 .< sum(get_demand_signals(false, [true,false], 0.0, 0.0) for i in 1:10000) .< 5500)
    @test get_demand_signals(false, [true,false], 0.5, 0.0) == [0, 0]
    @test get_demand_signals(false, [true,true], 0.0, 0.5) == [0, 0]
    @test get_demand_signals(true, [true,true], 0.0, 0.5) == [1, 1]
end

@testset "get_demand_level" begin
    @test get_demand_level(1.0) == true
    @test get_demand_level(1.0) == false
end
