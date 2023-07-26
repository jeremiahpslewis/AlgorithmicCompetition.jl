# """
#     Base.push!(multiagent::MultiAgentPolicy, ::PostActStage, env::DDDCEnv)

# Pushes the reward and terminal state of each player to their respective cache, updates the trajectory based on the cache.
# """
# function Base.push!(multiagent::MultiAgentPolicy, ::PostActStage, env::DDDCEnv)
#     for player in players(env)
#         agent = multiagent[player]
#         cache = agent.cache
#         push!(agent.trajectory, reward(env, player), is_terminated(env))
#         push!(agent.trajectory, cache, state(env, player))
#     end
# end

# """
#     RLBase.plan!(agent::Agent{P,T}, env::DDDCEnv, p::Symbol)

# Chooses an action for the agent based on the policy and pushes it to the cache. Returns the action.
# """
# function RLBase.plan!(agent::Agent{P,T}, env::DDDCEnv, p::Symbol) where {P,T}
#     action = RLBase.plan!(agent.policy, env, p)
#     push!(agent.cache, action)
#     action
# end
