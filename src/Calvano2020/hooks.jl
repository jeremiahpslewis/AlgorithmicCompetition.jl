using ReinforcementLearning

function get_best_action(d::Dict{Int8, BitSet}, state::Int16)
    for (k, v) in d
        if state in v
            return k
        else
            return Int8(0)
        end
    end
end

function update_best_action!(d::Dict{Int8, BitSet}, state::Int16, prev_best_action::Int8, best_action::Int8)
    delete!(d[prev_best_action], state)
    push!(d[best_action], state)
end

mutable struct ConvergenceMeta
    convergence_duration::Int32
    convergence_metric::Int32
    iterations_until_convergence::Int32
end

struct ConvergenceCheck <: AbstractHook
    best_response_lookup::Tuple{Dict{Int8, BitSet}, Dict{Int8, BitSet}}
    # Number of steps where no change has happened to argmax
    convergence_meta_tuple::Vector{ConvergenceMeta}

    function ConvergenceCheck(n_prices::Int, n_players::Int)
        new(
            (Dict(Int8(i) => BitSet(Int16(0)) for i in 0:n_prices), Dict(Int8(i) => BitSet(Int16(0)) for i in 0:n_prices)),
            [ConvergenceMeta(0, 0, 0), ConvergenceMeta(0, 0, 0)],
        )
    end
end

function update!(
    h::ConvergenceCheck,
    current_player_id::Int,
    state::Int16,
    best_action::Int8,
    prev_best_action::Int8,
)
    # Increment duration whenever argmax action is stable (convergence criteria)
    # Increment convergence metric (e.g. convergence not reached)
    # Keep track of number of iterations it takes until convergence

    is_converged = prev_best_action == best_action
    h.convergence_meta_tuple[current_player_id].iterations_until_convergence += 1

    if is_converged
        h.convergence_meta_tuple[current_player_id].convergence_duration += 1
        h.convergence_meta_tuple[current_player_id].convergence_metric += 1
    end

    # Update argmax matrix
    
    if ~is_converged
        update_best_action!(h.best_response_lookup[current_player_id], state, prev_best_action, best_action)
    end

end


function (h::ConvergenceCheck)(::PostActStage, policy, env)
    # Convergence is defined over argmax action for each state
    # E.g. best / greedy action
    current_player_id = env.current_player_idx
    # Only update best action for the state space which was played
    if policy.policy.name != current_player_id
        return
    end

    state = convert(Int16, RLBase.state(env))
    best_action = convert(Int8, argmax(@view policy.policy.policy.learner.approximator.table[:, state]))
    prev_best_action = get_best_action(h.best_response_lookup[current_player_id], state)

    update!(
        h,
        current_player_id,
        state,
        best_action,
        prev_best_action,
    )

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
