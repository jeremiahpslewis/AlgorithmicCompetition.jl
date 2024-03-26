using ReinforcementLearning
import ReinforcementLearning: RLCore

struct StopWhenConverged <: AbstractStopCondition end

"""
    RLCore.check_stop(s::StopWhenConverged, agent, env)

Returns true if the environment has converged for all players.
"""
function RLCore.check!(s::StopWhenConverged, agent, env)
    # false until converged, then true
    return all(env.convergence_vect)
end

"""
    AIAPCStop(env::AIAPCEnv; stop_on_convergence = true)

Returns a stop condition that stops when the environment has converged for all players.
"""
function AIAPCStop(env::E; stop_on_convergence::Bool = true) where {E<:AbstractEnv}
    stop_conditions = []
    push!(stop_conditions, StopAfterNEpisodes(env.max_iter, is_show_progress = false))
    if stop_on_convergence
        stop_converged = StopWhenConverged()
        push!(stop_conditions, stop_converged)
    end
    return StopIfAny(stop_conditions...)
end
