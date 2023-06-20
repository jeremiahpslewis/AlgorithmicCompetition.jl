using AlgorithmicCompetition:
    AlgorithmicCompetition,
    CompetitionParameters,
    AIAPCHyperParameters,
    AIAPCEnv,
    CompetitionSolution,
    Experiment,
    get_ϵ,
    find_all_max
using ReinforcementLearningCore: RLCore

α = Float64(0.15)
β = Float64(4e-6)
δ = 0.95
ξ = 0.1
n_prices = 15
max_iter = Int(1e6)
price_index = 1:n_prices

const player_lookup = (; Symbol(1) => 1, Symbol(2) => 2)

competition_params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))

competition_solution = CompetitionSolution(competition_params)

hyperparams = AIAPCHyperParameters(
    α,
    β,
    δ,
    max_iter,
    competition_solution;
    convergence_threshold = Int(1e5),
)

env = AIAPCEnv(hyperparams)
experiment = Experiment(env; stop_on_convergence = true)

function run_manual(experiment)
    si_vect = Int64[]
    spi_vect = Int64[]
    convergence = Int64[0, 0]
    best_responses = [zeros(Int64, 225), zeros(Int64, 225)]
    t = 0

    while t < max_iter
        t += 1
        if t == 1
            si_vect = rand(1:15, 2)
        else
            si_vect = copy(spi_vect) # Next state from last round is current state in this round
            spi_vect = Int64[]
        end
        
        state_int = experiment.env.state_space_lookup[si_vect...]

        # Act!
        for player_ in [Symbol(1), Symbol(2)]
            ϵ_threshold = get_ϵ(experiment.policy[player_].policy.explorer, t)
            if rand() < ϵ_threshold
                push!(spi_vect, rand(1:15)) # Overwrite next state
            else
                # NOTE: lazy argmax, could include tie breaking logic
                values = experiment.policy[player_].policy.learner.approximator.table[:, state_int]
                max_vals = find_all_max(values)[2]
                best_action = rand(max_vals)
                push!(spi_vect, best_action) # Overwrite next state
            end
        end

        statepi_int = experiment.env.state_space_lookup[spi_vect...]

        # Update!
        for player_ in [Symbol(1), Symbol(2)]
            player_int = player_lookup[player_]
            action_player = spi_vect[player_int] # Current action (next state, e.g. spi, is same as current action)
            old_q = experiment.policy[player_].policy.learner.approximator.table[action_player, state_int]
            # Profit from current actions, e.g. next state, spi
            profit_ = experiment.env.profit_array[spi_vect[1], spi_vect[2], player_int]
            # Max q over all actions from next state:
            max_q_spi = maximum(experiment.policy[player_].policy.learner.approximator.table[:, statepi_int])
            # Best response for !current! state
            values = experiment.policy[player_].policy.learner.approximator.table[:, state_int]
            max_vals = RLCore.find_all_max(values)[2]
            best_response_q = rand(max_vals)

            # q_value_updated = α * (π_ + γ * maximum(Q(app, s_plus_one)) - Q(app, s, a))
            q_delta = α * (profit_ + δ * maximum(max_q_spi) - old_q)
            experiment.policy[player_].policy.learner.approximator.table[action_player, state_int] += q_delta

            if best_responses[player_int][state_int] == best_response_q
                convergence[player_int] += 1
            else
                best_responses[player_int][state_int] = best_response_q
            end
        end

        if all(convergence .>= 1e5)
            break
        end
    end
    return experiment, convergence, t
end

experiment, convergence, t = run_manual(experiment)
summary = AlgorithmicCompetition.economic_summary(experiment)
profit_gain = AlgorithmicCompetition.profit_gain(summary.convergence_profit, experiment.env)
t

profit_gain

# 4.070906600373279
# 4.61282553058293
# maximum(experiment.policy[player_].policy.learner.approximator.table)
