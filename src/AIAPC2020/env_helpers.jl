"""
    construct_state_space_lookup(action_space, n_prices)

Construct a lookup table from action space to the state space.
"""
function construct_AIAPC_state_space_lookup(action_space, n_prices)
    @assert length(action_space) == n_prices^2
    state_space_lookup = reshape(Int16.(1:length(action_space)), n_prices, n_prices)
    return state_space_lookup
end


"""
    construct_AIAPC_profit_array(price_options, params, n_players)

Construct a 3-dimensional array which holds the profit for each player given a price pair.
The first dimension is player 1's action, the second dimension is player 2's action, and
the third dimension is the player index for their profit.
"""
function construct_AIAPC_profit_array(
    price_options::SVector{15,Float64},
    competition_params_dict::Dict{Symbol,CompetitionParameters},
    n_players::Int;
    demand_mode = :high,
)
    n_prices = length(price_options)


    params_ = competition_params_dict[demand_mode]

    profit_array = zeros(Float64, n_prices, n_prices, n_players)
    for k = 1:n_players
        for i = 1:n_prices
            for j = 1:n_prices
                profit_array[i, j, k] = π(price_options[i], price_options[j], params_)[k]
            end
        end
    end

    return profit_array
end
