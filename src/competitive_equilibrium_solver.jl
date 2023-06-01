# Parameters from pg 3374 AIAPC 2020
using JuMP
using Chain
using Ipopt
using Flux
using StaticArrays
using Statistics

struct CompetitionParameters
    μ::Float64
    a_0::Float64
    a::Tuple{Float64,Float64}
    c::Tuple{Float64,Float64}
    n_firms::Int64

    function CompetitionParameters(μ, a_0, a, c)
        length(a) != length(c) &&
            throw(DimensionMismatch("a and c must be the same length."))
        n_firms = length(a)
        new(μ, a_0, a, c, n_firms)
    end
end

function Q(p1, p2, params::CompetitionParameters)
    # Logit demand function from pg 3372 AIAPC 2020
    a_ = (params.a_0, params.a...)
    p_ = (0, p1, p2)
    mu_vect = ((a_ .- p_) ./ params.μ)
    q_out = softmax([mu_vect...])
    return q_out[2:3]
end

function π(p1::T, p2::T, params::CompetitionParameters) where {T<:Real}
    # Returns profit due to p_1
    q_ = Q(p1, p2, params)
    π_ = ((p1, p2) .- params.c) .* q_
    return π_
end

function p_BR(p_minus_i_::T, params::CompetitionParameters) where {T<:Real}
    # Best response Bertrand price
    π_i_(p_i_, p_minus_i_) = π(p_i_, p_minus_i_, params)[1]
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    register(model, :π_i_, 2, π_i_; autodiff = true)
    @variable(model, p_minus_i)
    @variable(model, p_i)
    @constraint(model, p_minus_i == p_minus_i_)
    @NLobjective(model, Max, π_i_(p_i, p_minus_i))

    optimize!(model)

    return value(p_i)
end

π_i(p_i::T, p_minus_i::T, params::CompetitionParameters) where {T<:Real} =
    π(p_i, p_minus_i, params)[1]
π_bertrand(p_1::T, params::CompetitionParameters) where {T<:Real} =
    π(p_1, p_BR(p_1, params), params)[1]
π_monop(p_1::T, p_2::T, params::CompetitionParameters) where {T<:Real} =
    sum(π(p_1, p_2, params)) / 2 # per-firm


function solve_monopolist(params::CompetitionParameters)
    model = Model(Ipopt.Optimizer)
    π_monop_(p_1, p_2) = π_monop(p_1, p_2, params)
    set_silent(model)
    register(model, :π_monop_, 2, π_monop_; autodiff = true)
    @variable(model, p_1)
    @variable(model, p_2)
    @NLobjective(model, Max, π_monop_(p_1, p_2))

    optimize!(model)

    return model, (value(p_1), value(p_2))
end


function solve_bertrand(params::CompetitionParameters)
    π_i_(p_1, p_2) = π_i(p_1, p_2, params)

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    register(model, :π_i_, 2, π_i_, autodiff = true)

    @variable(model, p_i)
    @NLparameter(model, p_min_i == 1)
    @NLobjective(model, Max, π_i_(p_i, p_min_i))

    optimize!(model)

    i = 0
    while !isapprox(value(p_i), value(p_min_i))
        i += 1
        set_value(p_min_i, value(p_i))
        optimize!(model)
    end

    return model, (value(p_i), value(p_min_i)), i
end

# function Q_0_i(p::T, p_vect::Vector{T}, δ::T) where {T<:Real} 
#     @chain p begin
#         π_i.(_, price_options)
#         mean()
#         _ / (1 - δ)
#     end
# end

# function Q_0(p_vect::Vector{T}, δ::T) where {T<:Real} 
#     Q_0_i.(p_vect, (p_vect,), δ)
# end

# TODO: Investigate and fix this???
# ν(m, n, k, β) = (m-1)^n / (m^((k*n) * (n + 1)) * (1 - exp(-β * (n+1))))
# ν(15, 2, 1, 2e-5)
