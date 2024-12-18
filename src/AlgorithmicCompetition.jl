module AlgorithmicCompetition

import Base: @invokelatest
using PrecompileTools: @compile_workload    # this is a small dependency


include("common.jl")
include("competitive_equilibrium_solver.jl")
include("AIAPC2020/AIAPC2020.jl")
include("DDDC2023/DDDC2023.jl")

@compile_workload begin
        run_dddc(
            n_parameter_iterations = 1,
            max_iter = Int(1e4),
            convergence_threshold = Int(1e2),
            n_grid_increments = 2,
            debug = false,
        )
end

end
