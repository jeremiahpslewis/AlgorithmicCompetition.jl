using YAML

include("a1_viz.jl")
include("dddc_viz.jl")

d_ = Dict(
    :n_simulations_aiapc => n_simulations_aiapc,
    :n_simulations_dddc => n_simulations_dddc,
)

YAML.write_file("plots/viz_metadata.yml", d_)
