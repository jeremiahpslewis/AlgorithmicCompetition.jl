using ReinforcementLearningCore

function AIAPCStop(env::AIAPCEnv; stop_on_convergence = true)
    stop_conditions = []
    push!(stop_conditions, StopAfterEpisode(env.max_iter, is_show_progress = false))
    if stop_on_convergence
        stop_converged = StopAfterNoImprovement(() -> env.convergence_int[1], 5)
        push!(stop_conditions, stop_converged)
    end
    return ComposedStopCondition(stop_conditions...)
end
