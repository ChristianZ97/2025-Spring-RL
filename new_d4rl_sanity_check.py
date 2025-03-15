#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 535514 Reinforcement Learning, HW1

The sanity check suggested by D4RL official repo
Source: https://github.com/Farama-Foundation/D4RL
Updated version: Migrating from D4RL to Gymnasium-Robotics and Minari
"""

import gymnasium as gym
import minari
import numpy as np

# Create the environment
env = gym.make('PointMaze-v0')  # Update to the corresponding Gymnasium-Robotics environment

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
# Load the dataset using Minari
dataset = minari.load_dataset('D4RL/maze2d-umaze-v1')
print(dataset.observations)

# Alternatively, use d4rl.qlearning_dataset which
# also adds next_observations.

# To mimic d4rl.qlearning_dataset, extract transitions and convert to dictionary format
transitions = dataset.get_transitions()
dataset = {
    'observations': np.array([t['observation'] for t in transitions]),
    'actions': np.array([t['action'] for t in transitions]),
    'rewards': np.array([t['reward'] for t in transitions]),
    'next_observations': np.array([t['next_observation'] for t in transitions]),
    'terminals': np.array([t['terminal'] for t in transitions])
}