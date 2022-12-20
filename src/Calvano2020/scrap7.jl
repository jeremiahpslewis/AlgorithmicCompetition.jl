using BenchmarkTools
using ReinforcementLearning
using Revise
using StaticArrays

struct ConvergenceMeta
    convergence_duration::Tuple{UInt16,UInt16}
    convergence_metric::Tuple{UInt16,UInt16}
    iterations_until_convergence::Tuple{UInt16,UInt16}
end

struct ConvergenceCheck <: AbstractHook
    approximator_table__state_argmax::AbstractMatrix{UInt8}
    # Number of steps where no change has happened to argmax
    convergence_meta_tuple::ConvergenceMeta

    function ConvergenceCheck(
        n_state_space::Int,
        n_players::Int,
    )
        new(
            (@SArray zeros(UInt8, n_players, n_state_space)),
            ConvergenceMeta((0,0),(0,0),(0,0)),
    )
    end
end


@btime ConvergenceCheck(225, 2)


# 69.133 ns (2 allocations: 624 bytes)
# 53.976 ns (2 allocations: 624 bytes)
# 80.098 ns (2 allocations: 624 bytes)
# 1.208 ns (0 allocations: 0 bytes)
# 22.523 ns (1 allocation: 496 bytes)

# 40.868 ns (2 allocations: 576 bytes)
# 22.272 ns (1 allocation: 496 bytes)
