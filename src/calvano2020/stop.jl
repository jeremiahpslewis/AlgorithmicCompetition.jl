using ReinforcementLearning

function CalvanoStop(convergence_check::ConvergenceCheck, n_iter::Int, convergence_threshold::Int)
    ComposedStopCondition(
        StopAfterEpisode(n_iter),
        StopAfterNoImprovement(
            () -> convergence_check.convergence_metric,
        convergence_threshold)
end
