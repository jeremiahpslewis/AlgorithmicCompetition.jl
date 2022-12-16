using ReinforcementLearning

function _convergence_check(q_table::Matrix{Float64}, convergence_table::SubArray{Int64}, state::Int)
    best_action = argmax(q_table[:, state])
    is_converged = convergence_table[state] == best_action

    return best_action, is_converged
end

struct ConvergenceCheck <: AbstractHook
    n_state_space::Int
    n_players::Int
    approximator_table__state_argmax::AbstractMatrix{Int}
    # Number of steps where no change has happened to argmax
    convergence_duration::Vector{Int}
    convergence_metric::Vector{Int}
    iterations_until_convergence::Vector{Int}

    function ConvergenceCheck(
        n_state_space::Int,
        n_players::Int,
    )
        new(
            n_state_space,
            n_players,
            (@SMatrix zeros(Int, n_players, n_state_space)),
            zeros(SVector{n_players, Int}),
            zeros(SVector{n_players, Int}),
            zeros(SVector{n_players, Int}),
    )
    end
end

function (h::ConvergenceCheck)(::PostActStage, policy, env)

    # Convergence is defined over argmax action for each state
    # E.g. best / greedy action
    current_player_id = env.current_player_idx
    # Only update best action for the state space which was played
    if policy.policy.name == current_player_id
        q_table = policy.policy.policy.learner.approximator.table
        convergence_table = @view h.approximator_table__state_argmax[current_player_id, :]
        state = RLBase.state(env)
        best_action, is_converged = _convergence_check(q_table, convergence_table, state)


        # Increment duration whenever argmax action is stable (convergence criteria)
        # Increment convergence metric (e.g. convergence not reached)
        # and update argmax matrix
        # Keep track of number of iterations it takes until convergence
        h.iterations_until_convergence[current_player_id] += 1

        if is_converged
            h.convergence_duration[current_player_id] += 1
        else
            h.convergence_duration[current_player_id] = 0
            h.approximator_table__state_argmax[current_player_id, state] = best_action
            h.convergence_metric[current_player_id] += 1
        end
    end
end

function CalvanoHook(env::AbstractEnv)
    MultiAgentHook(
        (
            p => ComposedHook(
                # TotalRewardPerEpisode(;is_display_on_exit=false),
                env.convergence_check
                ) for
            p in players(env)
        )...,
    )
end
