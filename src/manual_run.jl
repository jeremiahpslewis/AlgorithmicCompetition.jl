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
using Statistics

α = Float64(0.15)
β = Float64(4e-6)
δ = 0.95
ξ = 0.1
n_prices = 15
max_iter = Int(1e6)
convergence_threshold = Int(1e5) #max_iter - 1
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
    convergence_threshold = convergence_threshold,
)

env = AIAPCEnv(hyperparams)
experiment = Experiment(env; stop_on_convergence = true)
# experiment.policy[Symbol(1)].policy.learner.approximator.table
function run_manual(experiment)
    si_vect = Int64[0, 0]
    spi_vect = Int64[0, 0]
    convergence = Int64[0, 0]
    best_responses = [zeros(Int64, 225), zeros(Int64, 225)]
    t = 0

    while t < experiment.env.max_iter
        t += 1
        if t == 1
            si_vect = rand(1:15, 2)
        else
            si_vect = copy(spi_vect) # Next state from last round is current state in this round
            spi_vect = Int64[0,0]
        end
        
        state_int = experiment.env.state_space_lookup[si_vect...]

        # Act!
        for player_ in [Symbol(1), Symbol(2)]
            player_int = player_lookup[player_]
            ϵ_threshold = get_ϵ(experiment.policy[player_].policy.explorer, t)
            if rand() < ϵ_threshold
                spi_vect[player_int] = rand(1:15) # Overwrite next state
            else
                values = experiment.policy[player_].policy.learner.approximator.table[:, state_int]
                max_vals = find_all_max(values)[2]
                best_action = rand(max_vals)
                spi_vect[player_int] = best_action # Overwrite next state
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

            new_q = (1 - experiment.env.α) * old_q + experiment.env.α * (profit_ + (experiment.env.δ * max_q_spi))
                
            experiment.policy[player_].policy.learner.approximator.table[action_player, state_int] = new_q

            if best_responses[player_int][state_int] == best_response_q
                convergence[player_int] += 1
            else
                best_responses[player_int][state_int] = best_response_q
            end
        end

        if all(convergence .>= experiment.env.convergence_threshold)
            break
        end
    end
    return experiment, convergence, t
end

profit_gain_ = []
duration = []
profit_gain_basic_ = []
for i in 1:10
    experiment, convergence, t = run_manual(experiment)
    summary_ = AlgorithmicCompetition.economic_summary(experiment)
    profit_gain = AlgorithmicCompetition.profit_gain(summary_.convergence_profit, experiment.env)

    profit_delta = maximum(experiment.env.profit_array) - minimum(experiment.env.profit_array)

    profit_gain_basic = (mean(summary_.convergence_profit) - minimum(experiment.env.profit_array)) / profit_delta
    

    if all(convergence .>= experiment.env.convergence_threshold)
        push!(profit_gain_, profit_gain)
        push!(duration, t)
        push!(profit_gain_basic_, profit_gain_basic)
    end
end

mean(profit_gain_)
mean(duration)



# 4.070906600373279
# 4.61282553058293
# maximum(experiment.policy[player_].policy.learner.approximator.table)

experiment.policy[Symbol(1)].policy.learner.approximator.table
