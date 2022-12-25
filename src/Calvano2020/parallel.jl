function run_pmap(
    α::Float64,
    β::Float64,
    δ::Float64,
    price_options::Vector{Float64},
    competition_params::CompetitionParameters,
    p_Bert_nash_equilibrium::Float64,
    p_monop_opt::Float64
    )
    o_ = runCalvano(
            α,
            β,
            δ,
            price_options,
            competition_params,
            p_Bert_nash_equilibrium,
            p_monop_opt,
        )
    # return o_
    return economic_summary(o_)
end
