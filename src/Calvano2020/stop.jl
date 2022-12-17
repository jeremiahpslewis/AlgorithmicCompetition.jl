using ReinforcementLearning

function CalvanoStop(env::CalvanoEnv)
    ComposedStopCondition(
        StopAfterEpisode(env.max_iter, is_show_progress=false),
        # TODO: For n_players > 3
        StopAfterNoImprovement(
            () -> env.convergence_check.convergence_meta_tuple[1].convergence_metric + env.convergence_check.convergence_meta_tuple[2].convergence_metric,
            env.convergence_threshold,
        ),
    )
end
