using ReinforcementLearning

function CalvanoStop(env::CalvanoEnv; stop_on_convergence=true)
    stop_conditions = []
    push!(stop_conditions, StopAfterEpisode(env.max_iter, is_show_progress = false))
    if stop_on_convergence
        stop_converged = StopAfterNoImprovement(
            () -> sum(env.convergence_metric),
            env.convergence_threshold,
        )
        push!(stop_conditions, stop_converged)
    end
    return ComposedStopCondition(stop_conditions...)
end
