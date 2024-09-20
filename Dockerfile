FROM julia:1.10

COPY . /

RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); using AlgorithmicCompetition'
