using ReinforcementLearning

CalvanoPolicy(env::CalvanoEnv, init_matrix::Matrix, params::CalvanoParams) = MultiAgentManager(
    (
        Agent(policy = NamedPolicy(
            p => QBasedPolicy(;
            learner=TDLearner(;
                # TabularQApproximator with specified init matrix
                approximator=TabularApproximator( # Renamed LinearApproximator on master branch
                    init_matrix,
                    Descent(params.α)
                ),
                # For param info: https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/f97747923c6d7bbc5576f81664ed7b05a2ab8f1e/src/ReinforcementLearningZoo/src/algorithms/tabular/td_learner.jl#L15
                method=:SARS,
                γ=params.δ,
                n=0,					
            ),
            explorer=EpsilonGreedyExplorer(1//Int(round(1/params.β))))
        ), trajectory=VectorSARTTrajectory(;
                state=Int,
                action=Union{Int, NoOp},
                reward=Float64,
                terminal=Bool))
        for p in players(env)
    )...
)
