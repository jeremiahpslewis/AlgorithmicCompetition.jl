using ReinforcementLearningCore
import ReinforcementLearningCore: RLCore

struct StopWhenConverged <: AbstractStopCondition end

"""
    RLCore.check_stop(s::StopWhenConverged, agent, env)

Returns true if the environment has converged for all players.
"""
function RLCore.check_stop(s::StopWhenConverged, agent, env)
    # false until converged, then true
    return all(values(env.convergence_dict))
end

"""
    AIAPCStop(env::AIAPCEnv; stop_on_convergence = true)

Returns a stop condition that stops when the environment has converged for all players.
"""
function AIAPCStop(env::AIAPCEnv; stop_on_convergence = true)
    stop_conditions = []
    push!(stop_conditions, StopAfterEpisode(env.max_iter, is_show_progress = false))
    if stop_on_convergence
        stop_converged = StopWhenConverged()
        push!(stop_conditions, stop_converged)
    end
    return ComposedStopCondition(stop_conditions...)
end
