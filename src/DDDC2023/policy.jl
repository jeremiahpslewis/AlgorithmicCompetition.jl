using ReinforcementLearningCore
using Flux

"""
    Q_i_0(env::DDDCEnv)

Calculate the Q-value for player i at time t=0, given the price chosen by player i and assuming random play over the price options of player -i.
"""
function Q_i_0(env::DDDCEnv)
    Float64[mean(mean(env.profit_array[:, :, 1, :], dims = 2), dims = 3) ./ (1 - env.δ)...]
end

"""
    InitMatrix(env::DDDCEnv, mode = "zero")

Initialize the Q-matrix for the AIAPC environment.
"""
function InitMatrix(env::DDDCEnv; mode = "zero")
    if mode == "zero"
        return zeros(env.n_prices, env.n_state_space)
    elseif mode == "baseline"
        opponent_randomizes_expected_profit = Q_i_0(env)
        return repeat(opponent_randomizes_expected_profit, 1, env.n_state_space)
    elseif mode == "constant"
        return fill(5, env.n_prices, env.n_state_space)
    else
        @assert false "Unknown mode"
    end
end

"""
    AIAPCPolicy(env::DDDCEnv; mode = "baseline")

Create a policy for the AIAPC environment, with symmetric agents, using a tabular Q-learner. Mode deterimines the initialization of the Q-matrix.
"""
function DDDCPolicy(env::DDDCEnv; mode = "baseline")
    dddc_policy = MultiAgentPolicy(
        NamedTuple(
            p => Agent(
                QBasedPolicy(;
                    learner = TDLearner(;
                        # TabularQApproximator with specified init matrix
                        approximator = TabularApproximator(
                            InitMatrix(env, mode = mode),
                            Descent(env.α),
                        ),
                        # For param info: https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/f97747923c6d7bbc5576f81664ed7b05a2ab8f1e/src/ReinforcementLearningZoo/src/algorithms/tabular/td_learner.jl#L15
                        method = :SARS,
                        γ = env.δ,
                        n = 0,
                    ),
                    explorer = AIAPCEpsilonGreedyExplorer(env.β * 1e-5),
                ),
                Trajectory(
                    CircularArraySARTSTraces(;
                        capacity = 1,
                        state = Int64 => (),
                        action = Int8 => (),
                        reward = Float64 => (),
                        terminal = Bool => (),
                    ),
                    DummySampler(),
                    InsertSampleRatioController(),
                ),
            ) for p in players(env)
        ),
    )

    return dddc_policy
end
