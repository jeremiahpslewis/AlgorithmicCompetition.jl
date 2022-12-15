using ReinforcementLearning

function CalvanoStop(env::CalvanoEnv)
    ComposedStopCondition(
        StopAfterEpisode(env.max_iter, is_show_progress=false),
        StopAfterNoImprovement(
            () -> env.convergence_check.convergence_metric,
            env.convergence_threshold,
        ),
    )
end
