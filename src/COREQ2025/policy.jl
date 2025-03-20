import Base
using ReinforcementLearning
using ReinforcementLearningFarm: EpsilonSpeedyExplorer

struct COREQPolicy
    signal_policy::MultiAgentPolicy
    defect_policy::MultiAgentPolicy
    compliance_policy::MultiAgentPolicy
end

Base.show(io::IO, p::COREQPolicy) = println(io, "COREQPolicy")

function CompliancePolicy(env; mode = "baseline") # TODO use type signature ::ComplianceEnv)
    policy = MultiAgentPolicy(
        PlayerTuple(
            p => Agent(
                QBasedPolicy(;
                    learner = TDLearner(
                        # TabularQApproximator with specified init matrix
                        TabularApproximator(InitMatrix(env, mode = mode)),
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
end

function COREQPolicy(env::DDDCEnv; mode = "baseline")
    signal_policy = DDDCPolicy(env; mode = mode)
    defect_policy = DDDCPolicy(env; mode = mode)
    compliance_policy = CompliancePolicy(env; mode = mode)

    return COREQPolicy(signal_policy, defect_policy, compliance_policy)
end

struct COREQHyperParameters
    α::Float64
    β::Float64
    δ::Float64
    max_iter::Int
    convergence_threshold::Int
    data_demand_digital_params::DDDCExperimentalParams
end

struct COREQEnv <: AbstractEnv # N is profit_array dimension
    α::Float64                              # Learning parameter
    β::Float64                              # Exploration parameter
    δ::Float64                              # Discount factor
    max_iter::Int                           # Maximum number of iterations
    convergence_threshold::Int              # Convergence threshold

    n_players::Int                          # Number of players

    signal_env::DDDCEnv
    defect_env::DDDCEnv
    compliance_env::AbstractEnv

    hyperparams::COREQHyperParameters

    function COREQEnv(p::COREQHyperParameters)
        DDDCEnv(
            DDDCHyperParameters(
                p.α,
                p.β,
                p.δ,
                p.max_iter,
                p.convergence_threshold,
                competition_solution_dict,
                p.data_demand_digital_params,
            ),
        )
        signal_env = DDDCEnv(p)
        defect_env = DDDCEnv(p)
        compliance_env = ComplianceEnv(p)

        new(
            p.α,
            p.β,
            p.δ,
            p.max_iter,
            p.convergence_threshold,
            2, # n_players
            signal_env,
            defect_env,
            compliance_env,
            p,
        )
    end
end

# tests...

α = Float64(0.125)
β = Float64(4e-1)
δ = 0.95
max_iter = Int(1e9)
convergence_threshold = Int(1e6)

data_demand_digital_params = DDDCExperimentalParams(
    weak_signal_quality_level = 0.99,
    strong_signal_quality_level = 0.995,
    signal_is_strong = [true, false],
    frequency_high_demand = 0.9,
)

hyperparams = COREQHyperParameters(
    α,
    β,
    δ,
    max_iter,
    convergence_threshold,
    data_demand_digital_params,
)

env = COREQEnv(hyperparams)

p = COREQPolicy(env)
