#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 535514 Reinforcement Learning, HW1

The sanity check suggested by D4RL official repo
Source: https://github.com/Farama-Foundation/D4RL
Updated version: Migrating from D4RL to Gymnasium-Robotics and Minari
"""

import gymnasium as gym
import minari

# Create the environment
env = gym.make('PointMaze-v0')  # Update to the corresponding Gymnasium-Robotics environment

# Reset the environment
env.reset(seed=42)

# Perform a random action
env.step(env.action_space.sample())

# Load the dataset using Minari
dataset_id = 'pointmaze-umaze-v1'  # Replace with the actual dataset ID
dataset = minari.load_dataset(dataset_id)

# Print dataset information to verify correct loading
print("Dataset ID:", dataset.dataset_id)
print("Number of episodes:", len(dataset.episodes))

# Close the environment
env.close()