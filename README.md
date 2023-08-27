# AlgorithmicCompetition.jl

[![DOI](https://zenodo.org/badge/570286360.svg)](https://zenodo.org/badge/latestdoi/570286360)

[![CI](https://github.com/jeremiahpslewis/AlgorithmicCompetition.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jeremiahpslewis/AlgorithmicCompetition.jl/actions/workflows/CI.yml)

Reinforcement learning models of algorithmic competition

## How to Run

```julia
import AlgorithmicCompetition
using Chain
using Statistics
using DataFrameMacros
using CSV
using ParallelDataTransfer
using Distributed

n_procs_ = 2 # update n processors

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

aiapc_results = AlgorithmicCompetition.run_aiapc()
```

For citations of works this project is based on, see `citations.bib`.
