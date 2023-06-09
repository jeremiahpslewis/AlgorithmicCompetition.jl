using ReinforcementLearningCore
using StaticArrays
using Flux

"""
    Q_i_0(price, price_options, δ, params)

Calculate the Q-value for player i at time t=0, given the price chosen by player i and assuming random play over the price options of player -i.
"""
function Q_i_0(price::Float64, price_options::SVector{15,Float64}, δ::Float64, params::CompetitionParameters)
    mean(π.((price,), price_options, (params,))[1]) ./ (1 - δ)
end

function Q_i_0(price_options::SVector{15,Float64}, δ::Float64, params::CompetitionParameters)
    Q_i_0.(price_options, (price_options,), δ, (params,))
end

function InitMatrix(n_prices, n_state_space; mode="zero")
    @assert mode == "zero" "Only baseline mode is supported"
    return zeros(n_prices, n_state_space)
end

function InitMatrix(price_options::SVector{15,Float64},
        n_state_space::Int64,
        δ::Float64,
        params::CompetitionParameters;
        mode="baseline"
    )
    @assert mode == "baseline" "Only baseline mode is supported"
    opponent_randomizes_expected_profit = Q_i_0(price_options, δ, params)
    return repeat(opponent_randomizes_expected_profit, 1, n_state_space)
end

function AIAPCPolicy(env::AIAPCEnv)
    aiapc_policy = MultiAgentPolicy(
        NamedTuple(
            p => Agent(
                QBasedPolicy(;
                    learner = TDLearner(;
                        # TabularQApproximator with specified init matrix
                        approximator = TabularApproximator(
                            InitMatrix(
                                env.price_options,
                                env.n_state_space,
                                env.δ,
                                env.competition_solution.params,
                                mode="baseline"
                            ),
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

