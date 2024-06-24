# AlgorithmicCompetition.jl

[![DOI](https://zenodo.org/badge/570286360.svg)](https://zenodo.org/badge/latestdoi/570286360)

[![CI](https://github.com/jeremiahpslewis/AlgorithmicCompetition.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jeremiahpslewis/AlgorithmicCompetition.jl/actions/workflows/CI.yml)

[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/7837/badge)](https://www.bestpractices.dev/projects/7837)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

Tools for structuring and scaling research into algorithmic competition.

Components:

- Reinforcement learning models of algorithmic competition

## How to Run

```julia
import AlgorithmicCompetition
using Chain
using Statistics
using DataFrameMacros
using CSV
using ParallelDataTransfer
using Distributed

n_procs_ = 2 # update number of parallel processes

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

## How to Run on PC2 Cluster

```bash
tmux
TASK=DDDC bash src/run_pc2.sh
```

## AI / LLM Usage Statement

This project uses [Github Copilot](https://github.com/features/copilot) and [Chat-GPT 3](https://chat.openai.com) to assist software development and optimize code performance.

