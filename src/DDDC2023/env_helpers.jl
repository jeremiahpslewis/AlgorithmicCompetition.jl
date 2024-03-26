"""
    construct_DDDC_state_space_lookup(action_space, n_prices)

Construct a lookup table from action space to the state space.
"""
function construct_DDDC_state_space_lookup(action_space, n_prices)
    @assert length(action_space) == n_prices^2 * 4
    state_space_lookup = reshape(Int64.(1:length(action_space)), n_prices, n_prices, 2, 2)
    return state_space_lookup
end


"""
    construct_DDDC_profit_array(price_options, params, n_players)

Construct a 3-dimensional array which holds the profit for each player given a price pair.
The first dimension is player 1's action, the second dimension is player 2's action, and
the third dimension is the player index for their profit.
"""
function construct_DDDC_profit_array(
    price_options::Vector{Float64},
    competition_params_dict::Dict{Symbol,CompetitionParameters},
    n_players::Int;
)
    n_prices = length(price_options)

    profit_array = zeros(Float64, n_prices, n_prices, n_players, 2)
    for l in [:high, :low]
        for k = 1:n_players
            for i = 1:n_prices
                for j = 1:n_prices
                    profit_array[i, j, k, demand_to_index[l]] =
                        π(price_options[i], price_options[j], competition_params_dict[l])[k]
                end
            end
        end
    end

    return profit_array
end
