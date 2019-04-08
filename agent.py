import math
import random
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Tensor = torch.Tensor


class Parameters(object):
    def __init__(self):
        self.num_episodes = 2000
        self.solve_score = 13

        self.replay_capacity = 100000
        self.batch_size = 64

        self.update_target_frequency = 100

        self.learning_rate = 0.001
        self.gamma = 0.99

        self.e_greedy = 1.0
        self.e_greedy_min = 0.01
        self.e_greedy_decay = 200

        self.double = False
        self.dueling = False


class ExperienceReplay(object):
    def __init__(self, params):
        self.params = params

        self.memory = []
        self.position = 0

    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)

        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position = (self.position + 1) % self.params.replay_capacity

    def sample(self):
        return zip(*random.sample(self.memory, self.params.batch_size))

    def __len__(self):
        return len(self.memory)


class NeuralNetwork(torch.nn.Module):
    def __init__(self, params, num_states, num_actions):
        super(NeuralNetwork, self).__init__()

        self.params = params
        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1_size = 64
        self.fc2_size = 64

        self.fc1 = torch.nn.Linear(num_states, self.fc1_size)
        self.fc2 = torch.nn.Linear(self.fc1_size, self.fc2_size)
        self.fc3 = torch.nn.Linear(self.fc2_size, self.num_actions)

        self.activation = torch.nn.ReLU()

    def forward(self, state):
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))

        return self.fc3(x)


class DuelingNeuralNetwork(torch.nn.Module):
    def __init__(self, params, num_states, num_actions):
        super(DuelingNeuralNetwork, self).__init__()

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
    def __init__(self, params, num_states, num_actions, memory):
        self.params = params
        self.num_states = num_states
        self.num_actions = num_actions
        self.memory = memory

        NN = NeuralNetwork
        if self.params.dueling:
            NN = DuelingNeuralNetwork

        self.nn = NN(self.params, self.num_states, self.num_actions).to(device)
        self.target_nn = NN(self.params, self.num_states, self.num_actions).to(device)

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.nn.parameters(), lr=self.params.learning_rate)

        self.total_steps = 0
        self.target_updated_count = 0

    def save_model(self, path):
        torch.save(self.nn.state_dict(), path)

    def load_model(self, path):
        self.nn.load_state_dict(torch.load(path))

    def get_epsilon(self, total_steps):
        epsilon = self.params.e_greedy_min + (self.params.e_greedy - self.params.e_greedy_min) * \
                  math.exp(-1. * total_steps / self.params.e_greedy_decay)
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
        if len(self.memory) < self.params.batch_size:
            return

        state, action, new_state, reward, done = self.memory.sample()

        state = Tensor(state).to(device)
        new_state = Tensor(new_state).to(device)

        reward = Tensor(reward).to(device)
        action = torch.LongTensor(action).to(device)
        done = Tensor(done).to(device)

        if self.params.double:
            new_state_indeces = self.nn(new_state).detach()
            max_new_state_indeces = torch.max(new_state_indeces, 1)[1]

            new_state_values = self.target_nn(new_state).detach()
            max_new_state_values = new_state_values.gather(1, max_new_state_indeces.unsqueeze(1)).squeeze(1)
        else:
            new_state_values = self.target_nn(new_state).detach()
            max_new_state_values = torch.max(new_state_values, 1)[0]

        target_value = reward + (1 - done) * self.params.gamma * max_new_state_values

        predicted_value = self.nn(state).gather(1, action.unsqueeze(1)).squeeze(1)

        loss = self.loss_function(predicted_value, target_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.target_updated_count % self.params.update_target_frequency == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())

        self.target_updated_count += 1