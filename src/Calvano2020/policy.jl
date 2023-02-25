using ReinforcementLearning

AIAPCPolicy(env::AIAPCEnv) = MultiAgentManager(
    (
        Agent(
            policy = NamedPolicy(
                p => QBasedPolicy(;
                    learner = TDLearner(;
                        # TabularQApproximator with specified init matrix
                        approximator = TabularApproximator( # Renamed LinearApproximator on master branch
                            env.init_matrix,
                            Descent(env.α),
                        ),
                        # For param info: https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/f97747923c6d7bbc5576f81664ed7b05a2ab8f1e/src/ReinforcementLearningZoo/src/algorithms/tabular/td_learner.jl#L15
                        method = :SARS,
                        γ = env.δ,
                        n = 0,
                    ),
                    explorer = EpsilonGreedyExplorer(Int(round(1 / env.β))),
                ),
            ),
            trajectory = VectorSARTTrajectory(;
                state = Int16,
                action = Union{Int8,NoOp},
                reward = Float32,
                terminal = Bool,
            ),
        ) for p in players(env)
    )...,
)
