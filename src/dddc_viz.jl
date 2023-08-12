using CairoMakie
using Chain
using DataFrameMacros
using AlgebraOfGraphics
using CSV
using DataFrames
using Statistics

file_name = "simulation_results_dddc_2023-08-11T22:51:15.158.csv"
df = DataFrame(CSV.File(file_name))

plt1 = @chain df begin
    @transform(:signal_quality_is_high = string(:signal_quality_is_high))
    data(_) * mapping(:frequency_high_demand, :π_bar, color=:signal_quality_is_high => nonnumeric, row=:signal_quality_is_high) * visual(Scatter)
end
draw(plt1)

df_summary = @chain df begin
    @groupby(:signal_quality_is_high, :frequency_high_demand)
    @combine(
        mean(:π_bar),
        mean(:iterations_until_convergence),
        mean(:profit_min),
        mean(:profit_max),
    )
end

plt2 = @chain df_summary begin
    data(_) * mapping(:frequency_high_demand, :π_bar_mean, color=:signal_quality_is_high => nonnumeric) * visual(Scatter)
end
draw(plt2)

plt3 = @chain df_summary begin
    data(_) * mapping(:frequency_high_demand, :iterations_until_convergence_mean, color=:signal_quality_is_high => nonnumeric) * visual(Scatter)
end
draw(plt3)


# TODO: Make this profit for low signal agent
plt4 = @chain df_summary begin
    data(_) * mapping(:frequency_high_demand, :profit_min_mean, color=:signal_quality_is_high => nonnumeric) * visual(Scatter)
end
draw(plt4)

# TODO: Make this profit for high signal agent
plt5 = @chain df_summary begin
    data(_) * mapping(:frequency_high_demand, :profit_max_mean, color=:signal_quality_is_high => nonnumeric) * visual(Scatter)
end
draw(plt5)


# TODO: Look into different levels of low signal quality, whether the 'drop-off' happens subtly or abruptly
