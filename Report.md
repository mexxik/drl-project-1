# Report

This report provides brief description of the algorithms and methods used to solve the task.

## Structure

The project consists of the following elements:
* `agent.py` - this file contains core classes that implement learning of an agent:
    * `Parameters` - container for all hyperparameters. Check a separate section about these below.
    * `ExperienceReplay` - responsible for storage and retrieval of state-action-reward tuples used for experience replay technique.
    * `NeuralNetwork` - PyTorch neural network with 2 hidden layers and Rectified Linear Unit (ReLU) as activation function.
    * `DuelingNeuralNetwork` dueling neural network with 2 hidden layers that split into 2 additional layers: advantage and value, and then combined as the output.
    * `Agent` - a main class, that contains all logic for learning and retrieving of actions. Check Agent section of this document for more details.
* `Navigation.ipynb` - a Notebook for training neural networks. It contains description on how to use it. Some basic benchmarking, like graphs, are includes there as well.
* `test.py` - a script that loads trained model and renders an agent gathering bananas.    

## Agent Class

This class is a core to learning. Its functionality implemented in the following functions:
* `get_epsilon()` - calculates epsilon based on specified hyperparameters, episodes processed count and exponential equation.
* `get_action()` - returns an action based on e-greedy policy.
* `optimize()` - does an actual learning and neural network training, depending on the method and hyperparameters. 
    
## Hyperparameters

Here are hyperparameters used in for the training:
* `num_episodes` - maximum number of episodes to process.
* `solve_score` - score goal to reach to considers the task solved.
* `replay_capacity` - maximum capacity of experience replay buffer.
* `batch_size` - number of state-action-reward tuple from experience replay buffer to train neural network on.
* `update_target_frequency` - specifies how after to update target network during double training.
* `learning_rate` - leaning rate used during optimization.
* `gamma` - discount rate.
* `e_greedy` - starting value for epsilon (1.0 means that all actions will be random).
* `e_greedy_min` - minimum value for epsilon during calculation.
* `e_greedy_decay` - parameters used for exponential epsilon calculation.
* `double` - specifies if secondary target network is used.
* `dueling` - specifies if dueling network is constructed to learning.