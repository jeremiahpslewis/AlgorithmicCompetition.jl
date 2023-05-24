using ReinforcementLearningCore, ReinforcementLearningBase
using StaticArrays
import Base.push!

mutable struct ConvergenceCheck <: AbstractHook
    convergence_duration::Int32
    iterations_until_convergence::Int32
    best_response_vector::MVector{225,Int} # state x action # TODO: Fix hardcoding of n_states
    is_converged::Bool
    convergence_threshold::Int64

    function ConvergenceCheck(convergence_threshold::Int64)
        new(0, 0, MVector{225,Int}(zeros(Int, 225)), false, convergence_threshold) # TODO: Fix hardcoding of n_states
    end
end

function Base.push!(
    h::ConvergenceCheck,
    state_::Int64,
    best_action::Int,
    iter_converged::Bool,
)
    # Increment duration whenever argmax action is stable (convergence criteria)
    # Increment convergence metric (e.g. convergence not reached)
    # Keep track of number of iterations it takes until convergence

    h.iterations_until_convergence += 1

    if iter_converged
        h.convergence_duration += 1

        if h.convergence_duration >= h.convergence_threshold
            h.is_converged = true
        end
    else
        h.convergence_duration = 0
        h.best_response_vector[state_] = best_action
        h.is_converged = false
    end

    return
end

function _best_action_lookup(state_, table)
    @views argmax(table[:, state_])
end

function Base.push!(
    h::ConvergenceCheck,
    table::Matrix{F},
    state_::S,
) where {S,F<:AbstractFloat}
    # Convergence is defined over argmax action for each state 
    # E.g. best / greedy action
    best_action = _best_action_lookup(state_, table)
    iter_converged = (@views h.best_response_vector[state_] == best_action)

    Base.push!(h, state_, best_action, iter_converged)

    return h.is_converged
end

function Base.push!(
    h::ConvergenceCheck,
    ::PostActStage,
    policy::P,
    env::E,
    player::Symbol,
) where {P<:AbstractPolicy,E<:AbstractEnv}
    state_ = RLBase.state(env, player)
    env.convergence_dict[player] =
        Base.push!(h, policy.policy.learner.approximator.table, state_)
    return
end

# TODO: Figure out why the hook results are identical for both players
function AIAPCHook(env::AbstractEnv)
    MultiAgentHook(
        NamedTuple(
            p => ComposedHook(
                TotalRewardPerEpisode(; is_display_on_exit = false),
                # TODO: MultiAgent version of TotalRewardPerEpisode / better player handling for hooks
                ConvergenceCheck(env.convergence_threshold),
            ) for p in players(env)
        ),
    )
end

function Base.push!(
    hook::MultiAgentHook,
    stage::AbstractStage,
    policy::MultiAgentPolicy,
    env::AIAPCEnv,
)
    for p in (Symbol(1), Symbol(2))
        Base.push!(hook[p], stage, policy[p], env, p)
    end
end
