import AlgorithmicCompetition
using Chain
using Statistics
using DataFrameMacros
using CSV
using ParallelDataTransfer
using Distributed

n_procs_ = 2
# n_procs_ = 6

_procs = addprocs(
    n_procs_,
    topology = :master_worker,
    exeflags = ["--threads=1", "--project=$(Base.active_project())"],
)

@everywhere begin
    using Pkg
    Pkg.instantiate()
    using AlgorithmicCompetition
end

AlgorithmicCompetition.run_apaic(; n_parameter_iterations=1, csv_out_path="", max_iter=Int(100))
