#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D4RL Dataset Format Viewer

A script to print the format details of D4RL datasets.

Usage Example:
    python d4rl_format.py maze2d-umaze-v1 hopper-medium-v2
"""
import gym
import d4rl
import numpy as np
import sys

def print_dataset_format(env_name):

    print("-" * 100)
    print(f"\nDataset format for {env_name}:")
    print()
    
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    
    print(f"Dataset type: {type(dataset)}")
    print(f"Dataset keys: {list(dataset.keys())}")
    print("\nStructure details:")
    
    for key in dataset.keys():
        data = dataset[key]
        print(f"{key}:")
        print(f"  Type: {type(data).__name__}")
        print(f"  Shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
    
    print("Environment spaces:")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")

    env.close()

def main():
    default_env = 'maze2d-umaze-v1'
    env_names = sys.argv[1:] or [default_env]
    
    for env_name in env_names:
        print_dataset_format(env_name)

if __name__ == "__main__":
    main()