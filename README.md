## About

This is a solution for Project 1 (Banana Navigation) of Udacity Deep Reinforcement Learning Nanodegree Program

The project is implemented in Python 3.5 and PyTorch 1.0.1.

## Installation

* Install dependencies: 
`pip install -r dependencies.txt`

* Make sure to install Unity on your system (follow Udacity instruction in the lesson)

* Get Unity Banana Environment for your OS and put it in the root directory of this projects (get download links from Udacity lesson)

## Structure

This task is solved using Deep Q-Network algorithm described in [this](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) paper.

Besides the basic Deep Q-Network approach, this solution includes [Double Q-Learning](https://arxiv.org/abs/1509.06461) and [Dueling Network](https://arxiv.org/abs/1511.06581) implementations.

For more details of the implementations of the algorithms, check Report.md.

## Training

This repository includes trained models for all 3 methods:
* navigation_basic.pth
* navigation_double.pth
* navigation_dueling.pth

These are PyTorch weights and could be loaded by the Agent for testing.

To train all three models use Navigation.ipynb and follow its instructions.

## Testing

For testing use test.py script. It accepts 2 arguments: `env` and `model`.

`env` is a name of Unity Environment specific to your OS (for example, `Banana.app` for MacOS).

`model` sets a method and its value could be:
* `basic` (for plain Deep Q-Network with experience replay and target network)
* `double` (for Double Q-Learning, that includes basic plus secondary network)
* `dueling` (for Dueling Network approach)

Example (for MacOS): `python test.py Banana.app double`

## Report

Check Report.md and Navigation.ipynb for more details of the implementations. 
