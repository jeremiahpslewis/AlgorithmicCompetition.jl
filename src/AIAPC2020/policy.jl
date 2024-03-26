using ReinforcementLearning
using ReinforcementLearningFarm: EpsilonSpeedyExplorer

"""
    Q_i_0(env::AIAPCEnv)

Calculate the Q-value for player i at time t=0, given the price chosen by player i and assuming random play over the price options of player -i.
"""
function Q_i_0(env::AIAPCEnv)
    Float64[mean(env.profit_array[:, :, 1], dims = 2) ./ (1 - env.δ)...]
end

"""
    InitMatrix(env::AIAPCEnv, mode = "zero")

Initialize the Q-matrix for the AIAPC environment.
"""
function InitMatrix(env::AIAPCEnv; mode = "zero")
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
    AIAPCPolicy(env::AIAPCEnv; mode = "baseline")

Create a policy for the AIAPC environment, with symmetric agents, using a tabular Q-learner. Mode deterimines the initialization of the Q-matrix.
"""
function AIAPCPolicy(env::AIAPCEnv; mode = "baseline")
    aiapc_policy = MultiAgentPolicy(
        PlayerTuple(
            p => Agent(
                QBasedPolicy(;
                    learner = TDLearner(
                        # TabularQApproximator with specified init matrix
                        TabularApproximator(
                            InitMatrix(env, mode = mode),
                        ),
                        # For param info: https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/f97747923c6d7bbc5576f81664ed7b05a2ab8f1e/src/ReinforcementLearningZoo/src/algorithms/tabular/td_learner.jl#L15
                        :SARS;
                        γ = env.δ,
                        α = env.α,
                        n = 0,
                    ),
                    explorer = EpsilonSpeedyExplorer(env.β * 1e-5),
                ),
                Trajectory(
                    CircularArraySARTSTraces(;
                        capacity = 1,
                        state = Int64 => (),
                        action = Int64 => (),
                        reward = Float64 => (),
                        terminal = Bool => (),
                    ),
                    DummySampler(),
                    InsertSampleRatioController(),
                ),
            ) for p in players(env)
        ),
    )

    return aiapc_policy
end
