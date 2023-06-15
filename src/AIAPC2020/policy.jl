using ReinforcementLearningCore
using StaticArrays
using Flux

"""
    Q_i_0(price, price_options, δ, params)

Calculate the Q-value for player i at time t=0, given the price chosen by player i and assuming random play over the price options of player -i.
"""
function Q_i_0(
    price::Float64,
    price_options::SVector{15,Float64},
    δ::Float64,
    params::CompetitionParameters,
)
    mean(π.((price,), price_options, (params,))[1]) ./ (1 - δ)
end

function Q_i_0(
    price_options::SVector{15,Float64},
    δ::Float64,
    params::CompetitionParameters,
)
    Q_i_0.(price_options, (price_options,), δ, (params,))
end

"""
    InitMatrix(env::AIAPCEnv, mode = "zero")

Initialize the Q-matrix for the AIAPC environment.
"""
function InitMatrix(env::AIAPCEnv; mode = "zero")
    if mode == "zero"
        return zeros(env.n_prices, env.n_state_space)
    elseif mode == "baseline"
        opponent_randomizes_expected_profit =
            Q_i_0(env.price_options, env.δ, env.competition_solution.params)
        return repeat(opponent_randomizes_expected_profit, 1, env.n_state_space)
    elseif mode == "constant"
        return fill(5, env.n_prices, env.n_state_space)
    else
        @assert false "Unknown mode"
    end
end

function AIAPCPolicy(env::AIAPCEnv)
    aiapc_policy = MultiAgentPolicy(
        NamedTuple(
            p => Agent(
                QBasedPolicy(;
                    learner = TDLearner(;
                        # TabularQApproximator with specified init matrix
                        approximator = TabularApproximator(
                            InitMatrix(env, mode = "baseline"),
                            Descent(env.α),
                        ),
                        # For param info: https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/f97747923c6d7bbc5576f81664ed7b05a2ab8f1e/src/ReinforcementLearningZoo/src/algorithms/tabular/td_learner.jl#L15
                        method = :SARS,
                        γ = env.δ,
                        n = 0,
                    ),
                    explorer = AIAPCEpsilonGreedyExplorer(env.β), # * 2 # TODO: Drop this hack / attempt to get conversion to behave
                ),
                Trajectory(
                    CircularArraySARSTraces(;
                        capacity = 1,
                        state = Int64 => (),
                        action = Int64 => (),
                        reward = Float64 => (),
                        terminal = Bool => (),
                    ),
                    DummySampler(),
                    InsertSampleRatioController(),
                ),
                ART{Int,Float64,Bool}(),
            ) for p in players(env)
        ),
    )

    return aiapc_policy
end
