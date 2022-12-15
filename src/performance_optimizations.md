# Performance Optimizations

- Benchmark current function!
- See if you can move from mutable to immutable param objects
- Run JET.jl


## Benchmark

| Current Timings |
| -- |
| 3.980870 seconds (15.92 M allocations: 1.574 GiB, 3.82% gc time, 83.05% compilation time: 0% of which was recompilation) |
| BenchmarkTools: 555.939 ms (240281 allocations: 790.91 MiB) |
| post opt round 1 |
| 124.788 ms (211847 allocations: 790.06 MiB) |
| 113.626 ms (211846 allocations: 790.06 MiB) |
| 124.227 ms (211599 allocations: 790.05 MiB) |
| 158.794 ms (211601 allocations: 790.05 MiB) |
| 122.588 ms (211625 allocations: 790.05 MiB) |
| 17.995 ms (207609 allocations: 17.38 MiB) |
| 18.179 ms (205611 allocations: 17.04 MiB) |
| 18.111 ms (205607 allocations: 17.04 MiB) |
| 17.791 ms (207607 allocations: 17.38 MiB) |
| 17.900 ms (205609 allocations: 17.04 MiB) |
