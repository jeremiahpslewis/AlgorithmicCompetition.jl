using ReinforcementLearning

function CalvanoStop(calvano_params::CalvanoParams)
    ComposedStopCondition(
        StopAfterEpisode(calvano_params.max_iter),
        StopAfterNoImprovement(
            () -> calvano_params.convergence_check.convergence_metric,
            calvano_params.convergence_threshold),
    )
end
