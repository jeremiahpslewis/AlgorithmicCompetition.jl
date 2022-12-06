Base.@kwdef mutable struct CalvanoParams
    α::Float64
    β::Float64
    δ::Float64
    n_players::Int
    memory_length::Int
    price_options::Base.AbstractVecOrTuple{Float64}
    max_iter::Int
    convergence_threshold::Int
    n_prices::Int
    price_index::Vector{Int}
    convergence_check::ConvergenceCheck
    init_matrix::Matrix{Float64}

    function CalvanoParams(; α::Float64, β::Float64, δ::Float64, n_players::Int, memory_length::Int, price_options::Base.AbstractVecOrTuple{Float64}, max_iter::Int, convergence_threshold::Int)
        n_prices = length(price_options)
        price_index = 1:n_prices
        convergence_check = ConvergenceCheck(n_state_space=n_prices, n_players=n_players)
        init_matrix = zeros(n_prices, n_prices)
        new(α, β, δ, n_players, memory_length, price_options, max_iter, convergence_threshold, n_prices, price_index, convergence_check, init_matrix)
    end
end
