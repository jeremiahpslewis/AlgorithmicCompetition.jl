using DataFrames
using AlgorithmicCompetition: AIAPCHyperParameters, AIAPCSummary, CompetitionParameters, CompetitionSolution, run_and_extract, extract_sim_results
using Distributed
using ProgressMeter
using Random
using Chain
using DataFrameMacros
using Statistics

function test_key_AIAPC_points(; n_parameter_iterations = 1000)
    competition_params = CompetitionParameters(0.25, 0, (2, 2), (1, 1))

    competition_solution = CompetitionSolution(competition_params)

    test_params = DataFrame(
        :α => [0.08, 0.2, 0.15],
        :β => [2, 0.25, 1] .* 1e5,
        :iter_min => [0, 1.5e5, 0.5e5],
        :iter_max => [5e5,10e5, 1.5e5],
        :Δ_π_bar_min => [0.7, 0.75, 0.8],
        :Δ_π_bar_max => [0.8, 0.85, 0.9],
    )
    hyperparameter_vect = AIAPCHyperParameters.(
            test_params[!, :α],
            test_params[!, :β],
            (0.95,),
            (Int(1e7),),
            (competition_solution,),
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
        @groupby(:α, :β)
        @combine(:Δ_π_bar = mean(:π_bar),
                :iterations_until_convergence = mean(:iterations_until_convergence))
        leftjoin(test_params, on=[:α, :β])
    end

    return df_summary
end


n_procs_ = 2 # up to 8 performance cores on m1 (7 workers + 1 main)

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

exp_df = test_key_AIAPC_points(; n_parameter_iterations=100)
rmprocs(_procs)

@chain exp_df begin
    @transform(:convergence_match = :Δ_π_bar_max > :Δ_π_bar > :Δ_π_bar_min,
               :profit_match = :iter_max > :iterations_until_convergence > :iter_min,
               )
    @select(:α, :β, :convergence_match, :profit_match, :iter_min, :iterations_until_convergence,  :iter_max)
end
