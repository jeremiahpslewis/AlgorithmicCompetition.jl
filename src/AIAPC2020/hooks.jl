using ReinforcementLearningCore, ReinforcementLearningBase
using StaticArrays

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

function update!(h::ConvergenceCheck, state_::Int16, best_action::Int, iter_converged::Bool)
    # Increment duration whenever argmax action is stable (convergence criteria)
    # Increment convergence metric (e.g. convergence not reached)
    # Keep track of number of iterations it takes until convergence

    h.iterations_until_convergence += 1

    if iter_converged
        h.convergence_duration += 1
    else
        (h.convergence_duration != 0) && (h.convergence_duration = 0)
        h.best_response_vector[state_] = best_action
    end

    if h.convergence_duration >= h.convergence_threshold
        h.is_converged = true
    end

    return
end


function update!(h::ConvergenceCheck, ::PostEpisodeStage, policy, env)
    # Convergence is defined over argmax action for each state 
    # E.g. best / greedy action
    n_prices = env.n_prices

    state_ = RLBase.state(env)
    best_action = argmax(@view policy.policy.learner.approximator.table[:, state_])
    iter_converged = (@views h.best_response_vector[state_] == best_action)

    update!(h, state_, best_action, iter_converged)

    # If not 'finally' converged, then increment
    if ~h.is_converged
        env.convergence_int[1] += 1
    end

    return
end

# TODO: Figure out why the hook results are identical for both players
function AIAPCHook(env::AbstractEnv)
    MultiAgentHook(
        NamedTuple(
            p => ComposedHook(
                TotalRewardPerEpisode(; is_display_on_exit = false),
                ConvergenceCheck(env.convergence_threshold),
            ) for p in players(env)
        )
    )
end

# # Type should be s::AbstractStage, but didn't match the signature
# # update!(hook::TotalRewardPerEpisode, s, policy, env) = nothing
# update!(hook::AbstractHook, s::AbstractStage, policy, env) = nothing


# function update!(hook::TotalRewardPerEpisode, s::PostEpisodeStage, policy, env)
#     hook(s, policy, env)
#     return
# end


# function update!(hook::TotalRewardPerEpisode, s::PostActStage, policy, env)
#     hook(s, policy, env)
#     return
# end

