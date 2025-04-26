# main
import sys
import gymnasium as gym
import numpy as np
import os
import time
import random
from collections import namedtuple
import torch
import torch.nn as nn
# import wandb
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from train import agent_interact, agent_update, agent_evaluate
from algo import DDPG
from utils import ReplayMemory, OUNoise, get_device, set_seed_and_env

device = get_device()
print(f"\n Using device {device}\n")
random_seed = 42
env_name = 'Pendulum-v1'

def main(
    env,
    gamma=0.9998,
    tau=0.005,
    noise_scale=1.2,
    lr_a=1e-3,
    lr_c=1e-4,
    batch_size=64,
    num_episodes=4000,
    render=False,
    save_model=True,
    writer=None
    ):

    torch.autograd.set_detect_anomaly(True)

	# Adjust for different environment    
    if writer is None:
        writer = SummaryWriter("./tb_record_pendulum")
    replay_size =  int(1e6)
    warm_up = int(2e4) # 100 episodes for exploration
    reward_scale = 1e-1 # 10% of original reward


    hidden_size = 256
    updates_per_step = 1

    ewma_reward = 0
    rewards = [0]
    ewma_reward_history = [0]
    total_numsteps = 0
    updates = 0

    agent = DDPG(env.observation_space.shape[0], env.action_space, gamma, tau, hidden_size, lr_a=lr_a, lr_c=lr_c)
    ounoise = OUNoise(env.action_space.shape[0])
    memory = ReplayMemory(replay_size)

    try:
        for i_episode in range(num_episodes):
            ounoise.scale = noise_scale * (1 - i_episode / num_episodes)
            # ounoise.scale = noise_scale
            ounoise.reset()

            total_numsteps = agent_interact(writer, env, agent, memory, ounoise, total_numsteps, warm_up, reward_scale)
            if len(memory) >= warm_up:
                updates = agent_update(writer, agent, memory, batch_size, total_numsteps, updates_per_step, updates)
            SOLVED = agent_evaluate(writer, env, agent, i_episode, rewards, ewma_reward_history, reward_scale)

            if SOLVED:
                print(f"\nSolved at episode {i_episode} with Running reward {ewma_reward_history[-1]}!!\n")
                if save_model: agent.save_model(env_name, '.pth')
                break

    except KeyboardInterrupt:
        print(f"\nKeyboardInterrupt detected. Saving model...\n")
        if save_model: agent.save_model(env_name + "_interrupt", '.pth')

    finally:
        env.close()
        writer.close()

    return {
        'last_rewards': rewards[-10:],
        'best_reward': max(rewards),
        'ewma_reward': ewma_reward_history[-1],
        'rewards': rewards
    }

if __name__ == '__main__':

    env = set_seed_and_env(random_seed, env_name)
    main(env)
