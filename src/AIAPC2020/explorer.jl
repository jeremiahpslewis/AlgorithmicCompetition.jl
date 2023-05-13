import ReinforcementLearningCore: RLCore
import ReinforcementLearningBase: RLBase

function RLCore.EpsilonGreedyExplorer(decay_steps::Int)
    EpsilonGreedyExplorer(kind = :exp, ϵ_init = 1, ϵ_stable = 0, decay_steps = decay_steps)
end

function RLBase.plan!(p::QBasedPolicy{L,Ex}, env::AIAPCEnv, player::Symbol) where {L,Ex}
    return RLBase.plan!(p.explorer, RLCore.estimate_reward(p.learner.approximator, RLBase.state(env)), env.price_index)
end
