using ReinforcementLearning
using StaticArrays

mutable struct ConvergenceCheck <: AbstractHook
    convergence_duration::Int32
    convergence_metric::Int32
    iterations_until_convergence::Int32
    best_response_vector#::MVector{225, Int8} # state x action # TODO: Fix hardcoding of n_states
    function ConvergenceCheck()
        new(0, 0, 0, MVector{225, Int8}(zeros(Int8, 225))) # TODO: Fix hardcoding of n_states
    end
end

function update!(
    h::ConvergenceCheck,
    state_::Int16,
    best_action::Int8,
    prev_best_action::Int,
)
    # Increment duration whenever argmax action is stable (convergence criteria)
    # Increment convergence metric (e.g. convergence not reached)
    # Keep track of number of iterations it takes until convergence

    is_converged = prev_best_action == best_action
    h.iterations_until_convergence += 1

    if is_converged
        h.convergence_duration += 1
        h.convergence_metric += 1
    else
        @inbounds h.best_response_vector[state_] = best_action
    end
end
 

function (h::ConvergenceCheck)(::PostEpisodeStage, policy, env)
    # Convergence is defined over argmax action for each state 
    # E.g. best / greedy action
    current_player_id = nameof(policy)
    n_prices = env.env.n_prices
    
    state_ = convert(Int16, RLBase.state(env))
    best_action = convert(Int8, argmax(@view policy.policy.policy.learner.approximator.table[:, state_]))
    prev_best_action = argmax(h.best_response_vector[state_])

    update!(
        h,
        state_,
        best_action,
        prev_best_action,
    )
    env.env.convergence_metric[current_player_id] = h.convergence_metric
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
