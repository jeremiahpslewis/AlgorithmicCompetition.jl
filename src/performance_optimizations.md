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


15.775 ms (186059 allocations: 16.66 MiB)
15.430 ms (186076 allocations: 16.66 MiB)
15.072 ms (185833 allocations: 16.63 MiB)
18.628 ms (208519 allocations: 16.97 MiB)
15.904 ms (185829 allocations: 16.63 MiB) # Int seems to help!
15.725 ms (185835 allocations: 16.63 MiB)
15.130 ms (185794 allocations: 15.40 MiB) # Int
15.456 ms (185787 allocations: 12.41 MiB)
15.283 ms (185782 allocations: 12.41 MiB)
15.081 ms (184958 allocations: 11.85 MiB)
15.484 ms (184960 allocations: 11.85 MiB)
15.769 ms (184936 allocations: 11.85 MiB)
16.473 ms (188974 allocations: 12.32 MiB) # Tuple instead of vect in hook
15.738 ms (184943 allocations: 11.85 MiB) # drop custom stop condition
11.940 ms (116933 allocations: 8.64 MiB) # drop custom hook
14.745 ms (184952 allocations: 11.85 MiB)
Fail ## Use column-based instead of row based index
Fail ## Use q-table view
15.470 ms (164960 allocations: 11.09 MiB) ## Simplify map_memory_to_state
drop stop condition

# Make mutable...
TODO:
- [x] Figure out why n_state_space is 5k not ~200 (memory 1 -> ~200, 2 -> 5k)
- [ ] Add tests



# new round of performance tuning

10 mio iterations -> 4 minutes

 1e6 rounds with hook  22.088 s (230879256 allocations: 10.29 GiB)
 1e6 rounds without custom hook 1.810 s (18688628 allocations: 819.79 MiB)
  1.861 s (18698670 allocations: 825.20 MiB)
  
21.831 s (228980420 allocations: 10.10 GiB)
21.897 s (228980169 allocations: 10.05 GiB)
21.617 s (228980170 allocations: 10.05 GiB)

2.132 s (22898696 allocations: 1.01 GiB) # Dropping the meta tuple assignment:
 ...so need to fix the custom hook!!!
21.148 s (203980035 allocations: 9.28 GiB) drop vectorization
21.616 s (227980158 allocations: 10.53 GiB)
21.946 s (228980298 allocations: 10.05 GiB)
20.889 s (203980129 allocations: 9.28 GiB)
22.657 s (240660005 allocations: 10.35 GiB)
21.545 s (228980633 allocations: 10.05 GiB)
22.265 s (228979920 allocations: 10.59 GiB)
21.267 s (228980556 allocations: 10.59 GiB)
21.761 s (228980508 allocations: 9.87 GiB)
22.346 s (227000637 allocations: 10.41 GiB)
26.222 s (244995266 allocations: 21.30 GiB)
24.850 s (209000527 allocations: 9.48 GiB)
22.930 s (209000527 allocations: 9.26 GiB)

20.441 s (182000522 allocations: 8.35 GiB)
22.034 s (182000522 allocations: 8.35 GiB)
21.302 s (184000540 allocations: 8.44 GiB)
21.267 s (184000523 allocations: 8.44 GiB)
20.131 s (184000474 allocations: 8.44 GiB)

23.618 s (203000500 allocations: 8.60 GiB)
24.456 s (208005856 allocations: 8.89 GiB) # gets worse if smaller ints are not used, but not significantly...
24.158 s (203000500 allocations: 8.57 GiB)
23.958 s (203000500 allocations: 8.47 GiB) # Float32s help a bit
24.012 s (203000406 allocations: 8.47 GiB)
23.594 s (203000638 allocations: 8.47 GiB) # Stop recalculating action space
21.765 s (185013689 allocations: 7.54 GiB) # Stop recalculating profits!
20.325 s (174013689 allocations: 7.47 GiB)
19.466 s (173014763 allocations: 7.44 GiB)
14.491 s (149013687 allocations: 6.66 GiB) # MVector instead of MMatrix
14.821 s (149017914 allocations: 6.67 GiB) # Drop circ call
14.700 s (149016255 allocations: 6.62 GiB) # use tuple instead of vector
14.916 s (149013687 allocations: 6.62 GiB) # Int16 state object
14.351 s (149013689 allocations: 6.62 GiB)
14.075 s (145013687 allocations: 6.38 GiB) # BEST SO FAR.
14.631 s (145013706 allocations: 6.38 GiB) # few steps back (type instability?)
14.253 s (145013689 allocations: 6.38 GiB)
14.360 s (145013687 allocations: 6.38 GiB)
14.588 s (145013687 allocations: 6.38 GiB)
14.425 s (145013687 allocations: 6.38 GiB)
16.287 s (147013688 allocations: 13.06 GiB) # cartesianaxis boondoggle
14.783 s (145013687 allocations: 6.38 GiB)   # back, but slower (.= assignment?)
14.229 s (145013688 allocations: 6.38 GiB) # yep, .= assignment is slower...
14.590 s (145013687 allocations: 6.38 GiB)
14.156 s (145013687 allocations: 6.38 GiB) # try dropping convert statements. It helps
13.823 s (143013687 allocations: 6.29 GiB) # dropped another convert statement
13.748 s (143013687 allocations: 6.29 GiB) # try sprinkling inbounds absolutely everywhere
14.018 s (143013687 allocations: 6.29 GiB) # try splatting memory vector. Fails.
14.086 s (143013687 allocations: 6.29 GiB) # try alternate update method. Fails.
14.691 s (143013687 allocations: 6.29 GiB)
14.296 s (143013687 allocations: 6.29 GiB) # clean assignment
13.972 s (143013687 allocations: 6.29 GiB)
13.788 s (143013687 allocations: 6.29 GiB)
