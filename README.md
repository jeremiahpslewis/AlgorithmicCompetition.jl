# AlgorithmicCompetition.jl

[![DOI](https://zenodo.org/badge/570286360.svg)](https://zenodo.org/badge/latestdoi/570286360)

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

## TODO List

- Figure out broken test
- Run AIAPC and reproduce data viz
- Drop ReinforcementLearning.jl, use only RLCore & RLBase; refactor RLZoo code, initially within AlgComp.jl package
