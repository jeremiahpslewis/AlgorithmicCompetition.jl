struct CalvanoParams
    α::Float
    β::Float
    δ::Float
    n_players::Int
    memory_length::Int
    price_options::Vector{Float64}
    max_iter::Int
    n_prices::Int = length(price_options)
    price_index::Vector{Int} = 1:n_prices
end
