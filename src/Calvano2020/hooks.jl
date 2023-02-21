using ReinforcementLearning
using StaticArrays
using Accessors

mutable struct ConvergenceCheck <: AbstractHook
    convergence_duration::Int32
    iterations_until_convergence::Int32
    best_response_vector#::MVector{225, Int8} # state x action # TODO: Fix hardcoding of n_states
    function ConvergenceCheck()
        new(0, 0, MVector{225, Int8}(zeros(Int8, 225))) # TODO: Fix hardcoding of n_states
    end
end

function update!(
    h::ConvergenceCheck,
    env::AbstractEnv,
    current_player_id,
    state_::Int16,
    best_action::Int8,
    iter_converged::Bool,
)
    # Increment duration whenever argmax action is stable (convergence criteria)
    # Increment convergence metric (e.g. convergence not reached)
    # Keep track of number of iterations it takes until convergence

    h.iterations_until_convergence += 1

    if iter_converged
        h.convergence_duration += 1
    else
        (h.convergence_duration != 0) && (h.convergence_duration *= 0)
        @inbounds h.best_response_vector[state_] = best_action
    end

    if h.convergence_duration >= env.env.convergence_threshold
        @set env.env.is_converged[current_player_id] = true
    end
end
 

function (h::ConvergenceCheck)(::PostEpisodeStage, policy, env)
    # Convergence is defined over argmax action for each state 
    # E.g. best / greedy action
    current_player_id = nameof(policy)
    n_prices = env.env.n_prices
    
    state_ = convert(Int16, RLBase.state(env))
    best_action = convert(Int8, argmax(@view policy.policy.policy.learner.approximator.table[:, state_]))
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
