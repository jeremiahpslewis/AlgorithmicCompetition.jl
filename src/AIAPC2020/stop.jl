using ReinforcementLearningCore

struct StopWhenConverged end

function (s::StopWhenConverged)(agent, env)
    return env.convergence_dict[:1] > env.convergence_threshold && env.convergence_dict[:2] > env.convergence_threshold
end

function AIAPCStop(env::AIAPCEnv; stop_on_convergence = true)
    stop_conditions = []
    push!(stop_conditions, StopAfterEpisode(env.max_iter, is_show_progress = false))
    if stop_on_convergence
        stop_converged = StopWhenDone()
        push!(stop_conditions, stop_converged)
    end
    return ComposedStopCondition(stop_conditions...)
end
