import Base
using ReinforcementLearning
using ReinforcementLearningFarm: EpsilonSpeedyExplorer

struct COREQPolicy
    signal_policy::MultiAgentPolicy
    defect_policy::MultiAgentPolicy
    compliance_policy::MultiAgentPolicy

    function COREQPolicy(env::COREQEnv)
        signal_policy = DDDCPolicy(env.signal_env)
        defect_policy = DDDCPolicy(env.defect_env)
        compliance_policy = CompliancePolicy(env.compliance_env)

        new(signal_policy, defect_policy, compliance_policy)
    end
end

Base.show(io::IO, ::MIME"text/plain", p::COREQPolicy) = print(io, "COREQPolicy")
Base.show(io::IO, ::MIME"text/plain", p::COREQEnv) = print(io, "COREQEnv")

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

struct COREQEnv <: AbstractEnv
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
        competition_params_dict = Dict(
            :low => CompetitionParameters(0.25, 0.25, (2, 2), (1, 1)),
            :high => CompetitionParameters(0.25, -0.25, (2, 2), (1, 1)), # Parameter values aligned with Calvano 2020 Stochastic Demand case
        )

        competition_solution_dict = Dict(
            d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low]
        )

        signal_env = DDDCEnv(
            DDDCHyperParameters(
                p.α,
                p.β,
                p.δ,
                p.max_iter,
                competition_solution_dict,
                p.data_demand_digital_params,
                convergence_threshold = p.convergence_threshold,
            ),
        )

        defect_dddc_params = DDDCExperimentalParams(
            weak_signal_quality_level = p.data_demand_digital_params.weak_signal_quality_level,
            strong_signal_quality_level = p.data_demand_digital_params.strong_signal_quality_level,
            signal_is_strong = p.data_demand_digital_params.signal_is_strong,
            frequency_high_demand = p.data_demand_digital_params.frequency_high_demand,
            trembling_hand_frequency = 1.0, # Only visit trembling hand state, thus ignoring the signal entirely
        )
        defect_env = DDDCEnv(
            DDDCHyperParameters(
                p.α,
                p.β,
                p.δ,
                p.max_iter,
                competition_solution_dict,
                defect_dddc_params,
                convergence_threshold = p.convergence_threshold,
            ),
        )

        # TODO: add compliance env
        compliance_env = defect_env
        # compliance_env = ComplianceEnv(p)

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
            p
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
