# function (agent::Agent)(stage::PreActStage, env::SequentialEnv{AIAPCEnv}, action)
#     optimise!(agent.trajectory, agent.policy, env, stage, action)
#     optimise!(agent.policy, agent.trajectory, env, stage)
# end

# function (agent::Agent)(stage::AbstractStage, env::SequentialEnv{AIAPCEnv})
#     optimise!(agent.trajectory, agent.policy, env, stage)
#     optimise!(agent.policy, agent.trajectory, env, stage)
# end
