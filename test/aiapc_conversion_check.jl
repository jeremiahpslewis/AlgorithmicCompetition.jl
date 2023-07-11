using DataFrames
using AlgorithmicCompetition:
    AIAPCHyperParameters,
    AIAPCSummary,
    CompetitionParameters,
    CompetitionSolution,
    run_and_extract,
    extract_sim_results,
    profit_gain,
    AIAPCEnv
using Distributed
using ProgressMeter
using Random
using Chain
using DataFrameMacros
using Statistics

function test_key_AIAPC_points(; n_parameter_iterations = 1000)
    competition_params_dict = Dict(
        :high => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
        :low => CompetitionParameters(0.25, 0, (2, 2), (1, 1)),
    )
    competition_solution_dict = Dict(d_ => CompetitionSolution(competition_params_dict[d_]) for d_ in [:high, :low])

    test_params = DataFrame(
        :α => [0.15, 0.08, 0.2, 0.15, 0.01, 0.04, 0.1, 0.25, 0.2],
        :β => [0.4, 2, 0.25, 1, 0.1, 0.2, 1.75, 0.1, 1]
        :iter_min => [0, 0, 1.5e6, 0.5e6, 0, 1.5e6, 0.2e6, 0.5e6, 0.5e6],
        :iter_max => [1e7, 0.7e6, 1e7, 1.1e6, 1e7, 1e7, 0.5e6, 0.8e6, 1.5e6],
        :Δ_π_bar_min => [0.75, 0.7, 0.75, 0.75, 0.6, 0.85, 0.5, 0.65, 0.55],
        :Δ_π_bar_max => [0.9, 0.8, 0.85, 0.9, 1, 0.95, 0.8, 0.75, 0.75],
    )
    hyperparameter_vect =
        AIAPCHyperParameters.(
            test_params[!, :α],
            test_params[!, :β],
            (0.95,),
            (Int(1e9),),
            (competition_solution_dict,),
        )
    exp_list_ = AIAPCSummary[]
    exp_list = @showprogress pmap(
        run_and_extract,
        shuffle(repeat(hyperparameter_vect, n_parameter_iterations));
        on_error = identity,
    )
    append!(exp_list_, exp_list)

    df = extract_sim_results(exp_list_)

    df_summary = @chain df begin
        @subset(all(:is_converged))
        @groupby(:α, :β)
        @combine(
            :Δ_π_bar = profit_gain(:π_bar, AIAPCEnv(hyperparameter_vect[1])),
            :iterations_until_convergence = mean(:iterations_until_convergence)
        )
        leftjoin(test_params, on = [:α, :β])
    end

    return df_summary, exp_list_
end


n_procs_ = 7 # up to 8 performance cores on m1 (7 workers + 1 main)

_procs = addprocs(
    n_procs_,
    topology = :master_worker,
    exeflags = ["--threads=1", "--project=$(Base.active_project())"],
)

@everywhere begin
    using Pkg
    Pkg.instantiate()
    using AlgorithmicCompetition: run_and_extract
end

exp_df, exp_list = test_key_AIAPC_points(; n_parameter_iterations = 10)
rmprocs(_procs)

exp_diagostic = @chain exp_df begin
    @transform(
        :profit_match = :Δ_π_bar_max > :Δ_π_bar > :Δ_π_bar_min,
        :convergence_match = :iter_max > :iterations_until_convergence > :iter_min,
        :convergence_status =
            :iterations_until_convergence > :iter_max ? "too slow" :
            :iterations_until_convergence < :iter_min ? "too fast" : "ok",
        :profit_status =
            :Δ_π_bar > :Δ_π_bar_max ? "high" : :Δ_π_bar < :Δ_π_bar_min ? "low" : "ok"
    )
    @select(
        :α,
        :β,
        :iterations_until_convergence =
            round(:iterations_until_convergence; digits = 0),
        :convergence_status,
        :Δ_π_bar,
        :profit_status
    )
end

exp_diagostic[1:8, :]
# @test all(exp_diagostic[!, :profit_match])
# @test all(exp_diagostic[!, :convergence_match])
