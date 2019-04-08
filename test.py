from unityagents import UnityEnvironment

model_path = "navigation_{}.pth".format(name)

env = UnityEnvironment(file_name="Banana.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=False)[brain_name]

num_states = len(env_info.vector_observations[0])
num_actions = brain.vector_action_space_size

agent = Agent(params, num_states, num_actions, None)
agent.load_model(model_path)

state = env_info.vector_observations[0]
score = 0
while True:
    action = agent.get_action(state)
    env_info = env.step(action)[brain_name]
    next_state = env_info.vector_observations[0]
    reward = env_info.rewards[0]
    done = env_info.local_done[0]
    score += reward
    state = next_state
    if done:  #
        break

print("test score: {}".format(score))

env.close()