import sys
import math
import torch
import random
import numpy as np

from collections import deque
from unityagents import UnityEnvironment


# ------- params ------------
num_episodes = 2000

replay_capacity = 100000
batch_size = 64

update_target_frequency = 100
# ---------------------------

# ------- hyper params ------
learning_rate = 0.001
gamma = 0.99

e_greedy = 1.0
e_greedy_min = 0.01
e_greedy_decay = 200
# ---------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Tensor = torch.Tensor


class ExperienceReplay(object):
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)

        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=64):
        return zip(*random.sample(self.memory, batch_size))

    def __len__(self):
        return len(self.memory)


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_states, num_actions):
        super(NeuralNetwork, self).__init__()

        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1_size = 64
        self.fc2_size = 64

        self.fc1 = torch.nn.Linear(num_states, self.fc1_size)
        self.advantage = torch.nn.Linear(self.fc1_size, self.num_actions)
        self.value = torch.nn.Linear(self.fc1_size, 1)

        self.activation = torch.nn.ReLU()

    def forward(self, state):
        x = self.activation(self.fc1(state))

        advantage_output = self.advantage(x)
        value_output = self.value(x)

        final_output = value_output + advantage_output - advantage_output.mean()

        return final_output


class Agent(object):
    def __init__(self, num_states, num_actions, memory):
        self.num_states = num_states
        self.num_actions = num_actions
        self.memory = memory

        self.nn = NeuralNetwork(self.num_states, self.num_actions).to(device)
        self.target_nn = NeuralNetwork(self.num_states, self.num_actions).to(device)

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.nn.parameters(), lr=learning_rate)

        self.total_steps = 0
        self.target_updated_count = 0

    def get_epsilon(self, total_steps):
        epsilon = e_greedy_min + (e_greedy - e_greedy_min) * \
                  math.exp(-1. * total_steps / e_greedy_decay)
        return epsilon

    def get_action(self, state):
        self.total_steps += 1

        random_number = torch.rand(1)[0]
        epsilon = self.get_epsilon(self.total_steps)

        if random_number > epsilon:
            with torch.no_grad():
                state = Tensor(state).to(device)
                action = self.nn(state)
                action = torch.max(action, 0)[1].item()
        else:
            action = np.random.randint(self.num_actions)

        return action

    def optimize(self):
        if len(self.memory) < batch_size:
            return

        state, action, new_state, reward, done = self.memory.sample(batch_size)

        state = Tensor(state).to(device)
        new_state = Tensor(new_state).to(device)

        reward = Tensor(reward).to(device)
        action = torch.LongTensor(action).to(device)
        done = Tensor(done).to(device)

        new_state_indeces = self.nn(new_state).detach()
        max_new_state_indeces = torch.max(new_state_indeces, 1)[1]

        new_state_values = self.target_nn(new_state).detach()
        max_new_state_values = new_state_values.gather(1, max_new_state_indeces.unsqueeze(1)).squeeze(1)

        target_value = reward + (1 - done) * gamma * max_new_state_values

        predicted_value = self.nn(state).gather(1, action.unsqueeze(1)).squeeze(1)

        loss = self.loss_function(predicted_value, target_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.target_updated_count % update_target_frequency == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())

        self.target_updated_count += 1


env = UnityEnvironment(file_name="Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]

num_states = len(env_info.vector_observations[0])
num_actions = brain.vector_action_space_size

memory = ExperienceReplay(replay_capacity)
agent = Agent(num_states, num_actions, memory)

last_rewards = deque(maxlen=100)
average_rewards = deque(maxlen=num_episodes)

for i_episode in range(num_episodes):
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    score = 0

    while True:
        action = agent.get_action(state)

        env_info = env.step(action)[brain_name]
        new_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward

        memory.push(state, action, new_state, reward, done)
        agent.optimize()

        state = new_state
        if done:
            last_rewards.append(score)

            average_reward = np.mean(last_rewards)
            average_rewards.append(average_reward)

            print("\r {}/{}: average score {:.2f}".format(i_episode, num_episodes, average_reward), end="")
            sys.stdout.flush()

            break

env.close()