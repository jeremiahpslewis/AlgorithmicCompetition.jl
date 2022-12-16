using ReinforcementLearning

function CalvanoStop(env::CalvanoEnv)
    ComposedStopCondition(
        StopAfterEpisode(env.max_iter, is_show_progress=false),
        StopAfterNoImprovement(
            () -> sum(env.convergence_check.convergence_metric) / 2,
            env.convergence_threshold,
        ),
    )
end
