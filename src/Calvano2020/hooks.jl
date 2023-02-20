using ReinforcementLearning

mutable struct ConvergenceMeta
    convergence_duration::Int32
    convergence_metric::Int32
    iterations_until_convergence::Int32
    best_response_int::Int
end

mutable struct ConvergenceCheck <: AbstractHook
    convergence_duration::Int32
    convergence_metric::Int32
    iterations_until_convergence::Int32
    best_response_int::Int

    function ConvergenceCheck()
        new(0, 0, 0, 0)
    end
end

function update!(
    h::ConvergenceCheck,
    current_player_id::Int,
    state_::Int16,
    n_prices::Int,
    best_action::Int8,
    prev_best_action_vect::Vector{Int8},
)
    # Increment duration whenever argmax action is stable (convergence criteria)
    # Increment convergence metric (e.g. convergence not reached)
    # Keep track of number of iterations it takes until convergence

    is_converged = prev_best_action_vect[state_] == best_action
    h.iterations_until_convergence += 1

    if is_converged
        h.convergence_duration += 1
        h.convergence_metric += 1
    else
        prev_best_action_vect[state_] = best_action
        h.best_response_int += 1000
        # map_vect_to_int(prev_best_action_vect, n_prices)
    end

end


function (h::ConvergenceCheck)(::PostEpisodeStage, policy, env)
    # Convergence is defined over argmax action for each state
    # E.g. best / greedy action
    current_player_id = nameof(policy)
    n_prices = env.env.n_prices
    
    state_ = convert(Int16, RLBase.state(env))
    best_action = convert(Int8, argmax(@view policy.policy.policy.learner.approximator.table[:, state_]))
    prev_best_action_vect = map_int_to_vect(h.best_response_int, n_prices, state_)

    update!(
        h,
        current_player_id,
        state_,
        n_prices,
        best_action,
        prev_best_action_vect,
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
