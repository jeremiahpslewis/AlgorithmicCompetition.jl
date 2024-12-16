FROM julia:1.11

RUN mkdir /depot
ENV JULIA_PATH=/usr/local/julia
ENV JULIA_DEPOT_PATH=/depot
ENV JULIA_PROJECT=/algcomp

COPY . /algcomp

RUN julia --project=/algcomp -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); using AlgorithmicCompetition'

RUN chmod 777 -R /depot/

ENV JULIA_DEPOT_PATH="/tmp/:${JULIA_DEPOT_PATH}"
