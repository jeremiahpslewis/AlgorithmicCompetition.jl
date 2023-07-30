using ReinforcementLearningCore, ReinforcementLearningBase
import Base.push!


"""
    ConvergenceCheck(convergence_threshold::Int64)

Hook to check convergence, as defined by the best response for each state being stable for a given number of iterations.
"""
mutable struct ConvergenceCheck <: AbstractHook # N is n-states
    convergence_duration::Int64
    iterations_until_convergence::Int64
    best_response_vector::Vector{Int8} # state x action # TODO: Fix hardcoding of n_states
    is_converged::Bool
    convergence_threshold::Int64

    function ConvergenceCheck(n_states::Int64, convergence_threshold::Int64)
        new(0, 0, Vector{Int8}(zeros(Int8, n_states)), false, convergence_threshold) # TODO: Fix hardcoding of n_states
    end
end

function Base.push!(
    h::ConvergenceCheck,
    state_::S,
    best_action::Int8,
    iter_converged::Bool,
) where {S<:Integer}
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

"""
    _best_action_lookup(state_, table)

Look up the best action for a given state in the q-value matrix
"""
function _best_action_lookup(state_, table)
    best_action = 1
    max_value = table[1, state_]

    for i in 2:size(table, 1)
        value = table[i, state_]
        if value > max_value
            max_value = value
            best_action = i
        end
    end

    return Int8(best_action)
end

function Base.push!(
    h::ConvergenceCheck,
    table::Matrix{F},
    state_::S,
) where {S<:Integer,F<:AbstractFloat}
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
    agent::Agent{P,T},
    env::E,
    player::Symbol,
) where {P<:AbstractPolicy,T<:Trajectory,E<:AbstractEnv}
    Base.push!(h, PostActStage(), agent.policy, env, player)
end

function Base.push!(
    h::ConvergenceCheck,
    ::PostActStage,
    policy::QBasedPolicy{L,Exp},
    env::E,
    player::Symbol,
) where {L<:AbstractLearner,Exp<:AbstractExplorer,E<:AbstractEnv}
    Base.push!(h, PostActStage(), policy.learner, env, player)
end

function Base.push!(
    h::ConvergenceCheck,
    ::PostActStage,
    learner::L,
    env::E,
    player::Symbol,
) where {L<:AbstractLearner,E<:AbstractEnv}
    Base.push!(h, PostActStage(), learner.approximator, env, player)
end

function Base.push!(
    h::ConvergenceCheck,
    ::PostActStage,
    approximator::TabularApproximator{S,A},
    env::E,
    player::Symbol,
) where {S,A,E<:AbstractEnv}
    Base.push!(h, PostActStage(), approximator.table, env, player)
end

function Base.push!(
    h::ConvergenceCheck,
    ::PostActStage,
    table::Matrix{F},
    env::E,
    player::Symbol,
) where {F<:AbstractFloat,E<:AbstractEnv}
    state_ = RLBase.state(env, player)
    player_index = player_lookup[player]
    env.convergence_vect[player_index] = Base.push!(h, table, state_)
    return
end

# TODO: Figure out why the hook results are identical for both players
function AIAPCHook(env::AbstractEnv)
    MultiAgentHook(
        NamedTuple(
            p => ComposedHook(
                # TotalRewardPerEpisode(; is_display_on_exit = false),
                TotalRewardPerEpisodeLastN(; max_steps = env.convergence_threshold + 100),
                # TODO: MultiAgent version of TotalRewardPerEpisode / better player handling for hooks
                ConvergenceCheck(env.n_state_space, env.convergence_threshold),
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
    @simd for p in (Symbol(1), Symbol(2))
        Base.push!(hook[p], stage, policy[p], env, p)
    end
end
