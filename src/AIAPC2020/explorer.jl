using ReinforcementLearningCore

function RLCore.EpsilonGreedyExplorer(decay_steps::Int)
    EpsilonGreedyExplorer(kind = :exp, ϵ_init = 1, ϵ_stable = 0, decay_steps = decay_steps)
end
