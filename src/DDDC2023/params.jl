function construct_DDDC_action_space(price_index)
    Tuple(CartesianIndex{4}(i, j, k, l) for i in price_index for j in price_index for k in 1:2 for l in 1:2)
end
