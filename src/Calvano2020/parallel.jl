function run_pmap(
    i::Int,
    param_set::Vector{Tuple{Float64, Float64}},
    δ::Float64,
    price_options::Vector{Float64},
    competition_params::CompetitionParameters,
    p_Bert_nash_equilibrium::Float64,
    p_monop_opt::Float64
    )
    o_ = runCalvano(
            param_set[i][1],
            param_set[i][2],
            δ,
            price_options,
            competition_params,
            p_Bert_nash_equilibrium,
            p_monop_opt,
        )
    return economic_summary(o_)
end
