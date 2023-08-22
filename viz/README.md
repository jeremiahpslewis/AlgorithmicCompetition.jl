# Reproducing Data, Demand, and Digital Competition Graphics

In order to reproduce the graphics in *Data, Demand, and Digital Competition*, please do the following:

1. Install [Julia `v1.10`](https://julialang.org).
2. Set your current directory in a terminal shell to the `viz/` folder of this project.
3. Open a new Julia session.
4. Activate the local environment, `using Pkg; Pkg.activate(); Pkg.instantiate()`
5. Run the file `viz/run_viz.jl`

If you would like to regenerate the data used in the visualizations, you need to install the `AlgorithmicCompetition` package located [here](https://github.com/jeremiahpslewis/AlgorithmicCompetition.jl), via `using Pkg; Pkg.add("https://github.com/jeremiahpslewis/AlgorithmicCompetition.jl")`. Then run either the `src/multiprocessing_template_aiapc.jl` or the `src/multiprocessing_template_dddc.jl` depending on which study you would like to reproduce, adjusting the number of cores and other parameters to match your system.

If you have questions, please ping me by creating an issue at the [Algorithmic Competition Github Repository](https://github.com/jeremiahpslewis/AlgorithmicCompetition.jl).
