using ReinforcementLearning
using StaticArrays
using Accessors

mutable struct ConvergenceCheck <: AbstractHook
    convergence_duration::Int32
    iterations_until_convergence::Int32
    best_response_vector::MVector{225, Int} # state x action # TODO: Fix hardcoding of n_states
    is_converged::Bool
    function ConvergenceCheck()
        new(0, 0, MVector{225, Int}(zeros(Int, 225)), false) # TODO: Fix hardcoding of n_states
    end
end

function update!(
    h::ConvergenceCheck,
    env::AbstractEnv,
    current_player_id,
    state_::Int,
    best_action::Int,
    iter_converged::Bool,
)
    # Increment duration whenever argmax action is stable (convergence criteria)
    # Increment convergence metric (e.g. convergence not reached)
    # Keep track of number of iterations it takes until convergence

    h.iterations_until_convergence += 1

    if iter_converged
        h.convergence_duration += 1
    else
        (h.convergence_duration != 0) && (h.convergence_duration = 0)
        @inbounds h.best_response_vector[state_] = best_action
    end

    # If not 'finally' converged, then increment
    if ~h.is_converged
        env.env.convergence_int[1] += 1
    end

    if h.convergence_duration == env.env.convergence_threshold
        h.is_converged = true
    end
end
 

function (h::ConvergenceCheck)(::PostEpisodeStage, policy, env)
    # Convergence is defined over argmax action for each state 
    # E.g. best / greedy action
    current_player_id = nameof(policy)
    n_prices = env.env.n_prices
    
    state_ = RLBase.state(env)
    best_action = argmax(@view policy.policy.policy.learner.approximator.table[:, state_])
    iter_converged = (@view h.best_response_vector[state_]) == best_action

    update!(
        h,
        env,
        current_player_id,
        state_,
        best_action,
        iter_converged,
    )
    return
end

# TODO: Figure out why the hook results are identical for both players
function CalvanoHook(env::AbstractEnv)
    MultiAgentHook(
        (
            p => ComposedHook(
                TotalRewardPerEpisode(; is_display_on_exit = false),
                ConvergenceCheck(),
            ) for p in players(env)
        )...,
    )
end
