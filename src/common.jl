import Base.push!
using ReinforcementLearning
using DrWatson

const player_to_index = Dict(Player(1) => 1, Player(2) => 2)
const demand_to_index = (; :high => 1, :low => 2)

# Handle CartesianIndex actions
function Base.push!(
    multiagent::MultiAgentPolicy,
    ::PostActStage,
    env::E,
    actions::CartesianIndex,
) where {E<:AbstractEnv}
    actions = Tuple(actions)
    Base.push!(multiagent, PostActStage(), env, actions)
end
