module AlgorithmicCompetition

import Base: @invokelatest
using PrecompileTools: @compile_workload


include("common.jl")
include("competitive_equilibrium_solver.jl")
include("AIAPC2020/AIAPC2020.jl")
include("DDDC2023/DDDC2023.jl")

@compile_workload begin
    @info "Start precompile script."
    run_dddc(
        n_parameter_iterations = 1,
        max_iter = Int(1e2),
        convergence_threshold = Int(1e1),
        n_grid_increments = 1,
        debug = false,
        precompile = true,
    )
    @info "Precompile script completed."
end

end
