using ReinforcementLearningCore

struct StopWhenConverged <: AbstractStopCondition end

function (s::StopWhenConverged)(agent, env)
    # false until converged, then true
    return (env.convergence_dict[Symbol(1)] && env.convergence_dict[Symbol(2)])
end

function AIAPCStop(env::AIAPCEnv; stop_on_convergence = true)
    stop_conditions = []
    push!(stop_conditions, StopAfterEpisode(env.max_iter, is_show_progress = false))
    if stop_on_convergence
        stop_converged = StopWhenConverged()
        push!(stop_conditions, stop_converged)
    end
    return ComposedStopCondition(stop_conditions...)
end
