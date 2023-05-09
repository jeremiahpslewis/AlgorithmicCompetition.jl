using ReinforcementLearningCore
using StaticArrays

function InitMatrix(n_prices, n_state_space)
    return zeros(Float32, n_prices, n_state_space)
end

AIAPCPolicy(env::AIAPCEnv) = MultiAgentPolicy(
    NamedTuple(
        p => Agent(
            QBasedPolicy(;
                learner = TDLearner(;
                    # TabularQApproximator with specified init matrix
                    approximator = TabularApproximator(
                        InitMatrix(env.n_prices, env.n_state_space),
                        Descent(env.α),
                    ),
                    # For param info: https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/f97747923c6d7bbc5576f81664ed7b05a2ab8f1e/src/ReinforcementLearningZoo/src/algorithms/tabular/td_learner.jl#L15
                    method = :SARS,
                    γ = env.δ,
                    n = 0,
                ),
                explorer = AIAPCEpsilonGreedyExplorer(Float32(1e-5)),
            ),
            Trajectory(
                CircularArraySARTTraces(;
                    capacity = 3,
                    state = Int64 => (225,),
                    action = Int64 => (15,),
                    reward = Float32 => (),
                    terminal = Bool => (),
                ),
                DummySampler(),
            ),
            SRT{Int64, Float32, Bool}()
        ) for p in players(env)
    ),
)
