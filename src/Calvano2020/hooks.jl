using ReinforcementLearning

struct ConvergenceMeta
    convergence_duration::UInt32
    convergence_metric::UInt32
    iterations_until_convergence::UInt32
end

struct ConvergenceCheck <: AbstractHook
    approximator_table__state_argmax::Matrix{UInt8}
    # Number of steps where no change has happened to argmax
    convergence_meta_tuple::Vector{ConvergenceMeta}

    function ConvergenceCheck(n_state_space::Int, n_players::Int)
        new(
            zeros(UInt8, n_state_space, n_players),
            [ConvergenceMeta(0, 0, 0), ConvergenceMeta(0, 0, 0)],
        )
    end
end

function calculate_convergence_meta(
    c_meta::ConvergenceMeta,
    best_action::UInt8,
    prev_best_action::UInt8,
)
    # Increment duration whenever argmax action is stable (convergence criteria)
    # Increment convergence metric (e.g. convergence not reached)
    # Keep track of number of iterations it takes until convergence

    is_converged = prev_best_action == best_action
    iterations_until_convergence = c_meta.iterations_until_convergence + 1

    convergence_duration = is_converged ? c_meta.convergence_duration + 1 : 0
    convergence_metric =
        is_converged ? c_meta.convergence_metric : c_meta.convergence_metric + 1

    return ConvergenceMeta(
        convergence_duration,
        convergence_metric,
        iterations_until_convergence,
    ),
    is_converged,
    best_action

end


function (h::ConvergenceCheck)(::PostActStage, policy, env)
    # Convergence is defined over argmax action for each state
    # E.g. best / greedy action
    current_player_id = env.current_player_idx
    # Only update best action for the state space which was played
    if policy.policy.name != current_player_id
        return
    end

    state = RLBase.state(env)
    best_action = argmax(@view policy.policy.policy.learner.approximator.table[state, :])
    prev_best_action = (@view h.approximator_table__state_argmax[state, current_player_id])[1]

    c_meta, is_converged, best_action = calculate_convergence_meta(
        h.convergence_meta_tuple[current_player_id],
        best_action,
        prev_best_action,
    )

    h.convergence_meta_tuple[current_player_id] = c_meta

    # Update argmax matrix
    if ~is_converged
        h.approximator_table__state_argmax[state, current_player_id] = best_action
    end
    return

end

# TODO: Figure out why the hook results are identical for both players
function CalvanoHook(env::AbstractEnv)
    MultiAgentHook(
        (
            p => ComposedHook(
                TotalRewardPerEpisode(; is_display_on_exit = false),
                env.convergence_check,
            ) for p in players(env)
        )...,
    )
end
