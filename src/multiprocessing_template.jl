import AlgorithmicCompetition
using Chain
using Statistics
using DataFrameMacros
using CSV
using ParallelDataTransfer
using Distributed

@testset "run multiprocessing code"
    n_procs_ = 7

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

    AlgorithmicCompetition.run_aiapc(; n_parameter_iterations=1, csv_out_path="")
end
