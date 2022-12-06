using ReinforcementLearning

CalvanoPolicy(env::CalvanoEnv) = MultiAgentManager(
    (
        Agent(policy = NamedPolicy(
            p => QBasedPolicy(;
            learner=TDLearner(;
                # TabularQApproximator with specified init matrix
                approximator=TabularApproximator( # Renamed LinearApproximator on master branch
                    env.params.init_matrix,
                    Descent(env.params.α)
                ),
                # For param info: https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/f97747923c6d7bbc5576f81664ed7b05a2ab8f1e/src/ReinforcementLearningZoo/src/algorithms/tabular/td_learner.jl#L15
                method=:SARS,
                γ=env.params.δ,
                n=0,					
            ),
            explorer=EpsilonGreedyExplorer(1//Int(round(1 / env.params.β))))
        ), trajectory=VectorSARTTrajectory(;
                state=Int,
                action=Union{Int, NoOp},
                reward=Float64,
                terminal=Bool))
        for p in players(env)
    )...
)
