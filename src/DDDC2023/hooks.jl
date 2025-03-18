using CircularArrayBuffers
using Tables

struct DDDCTotalRewardPerLastNEpisodes <: AbstractHook
    rewards::CircularVectorBuffer{Float64}
    demand_state_high_vect::CircularVectorBuffer{Bool}

    function DDDCTotalRewardPerLastNEpisodes(; max_steps = 100)
        new(CircularVectorBuffer{Float64}(max_steps), CircularVectorBuffer{Bool}(max_steps))
    end
end

function Base.push!(h::DDDCTotalRewardPerLastNEpisodes, reward::Float64, memory::DDDCMemory)
    push!(h.rewards, reward)
    push!(h.demand_state_high_vect, memory.demand_state == :high)
    return
end

function Base.push!(
    h::DDDCTotalRewardPerLastNEpisodes,
    ::PostActStage,
    agent::P,
    env::DDDCEnv,
    player::Player,
) where {P<:AbstractPolicy}
    push!(h, reward(env, player), env.memory)
    return
end

function Base.push!(
    hook::DDDCTotalRewardPerLastNEpisodes,
    stage::Union{PreEpisodeStage,PostEpisodeStage,PostExperimentStage},
    agent::P,
    env::DDDCEnv,
    player::Player,
) where {P<:AbstractPolicy}
    return
end

function Base.push!(
    hook::MultiAgentHook,
    stage::AbstractStage,
    policy::MultiAgentPolicy,
    env::DDDCEnv,
)
    @simd for p in (Player(1), Player(2))
        push!(hook[p], stage, policy[p], env, p)
    end
end

struct DDDCPricesPerLastNEpisodes <: AbstractHook
    prices::CircularVectorBuffer{Int}

    function DDDCPricesPerLastNEpisodes(; max_steps=100)
        new(CircularVectorBuffer{Int}(max_steps))
    end
end

function Base.push!(h::DDDCPricesPerLastNEpisodes, price::Int)
    push!(h.prices, price)
    return
end

function Base.push!(h::DDDCPricesPerLastNEpisodes, ::PostActStage, agent::P, env::DDDCEnv, player::Player) where {P<:AbstractPolicy}
    # Replace `chosen_price(env, player)` with the actual function extracting the chosen price.
    push!(h, agent.trajectory.container[:action][end])
    return
end

function Base.push!(h::DDDCPricesPerLastNEpisodes, stage::Union{PreEpisodeStage, PostEpisodeStage, PostExperimentStage}, agent::P, env::DDDCEnv, player::Player) where {P<:AbstractPolicy}
    return
end

function DDDCHook(env::AbstractEnv)
    MultiAgentHook(
        PlayerTuple(
            p => ComposedHook(
                ConvergenceCheck(env.n_state_space, env.convergence_threshold),
                DDDCTotalRewardPerLastNEpisodes(;
                    max_steps = env.convergence_threshold + 100,
                ),
                # DDDCPricesPerLastNEpisodes(;
                #     max_steps = 100,
                # ),
            ) for p in players(env)
        ),
    )
end

Tables.istable(::Type{DDDCPricesPerLastNEpisodes}) = true
Tables.columnaccess(::Type{DDDCPricesPerLastNEpisodes}) = true
Tables.columns(h::DDDCPricesPerLastNEpisodes) = (; prices = h.prices)

# Make the hook table compatible with Tables.jl / accessible as DataFrame
Tables.istable(::Type{DDDCTotalRewardPerLastNEpisodes}) = true
Tables.columnaccess(::Type{DDDCTotalRewardPerLastNEpisodes}) = true
Tables.columns(h::DDDCTotalRewardPerLastNEpisodes) =
    (; rewards = h.rewards, demand_state_high_vect = h.demand_state_high_vect)

    