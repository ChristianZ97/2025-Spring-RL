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
    lr_c=3e-3, # tuned via BO
    num_episodes=2000,
    render=False,
    save_model=True,
    writer=None
    ):

    torch.autograd.set_detect_anomaly(True)

    # Tuned hyper
    lr_a=1.2e-3
    gamma=0.9998
    tau=0.025
    noise_scale=1.2
    batch_size=64

    '''
    Main Hyperparameters for DDPG

    gamma:        Discount factor determining the importance of future rewards.
    Higher values emphasize long-term rewards, while lower values make the agent favor immediate rewards.

    tau:          Soft update rate for synchronizing the target networks.
    Smaller values make updates more stable but slower; larger values speed up updates but can destabilize training.

    noise_scale:  Scaling factor for the Ornstein-Uhlenbeck noise applied to actions.
    Higher noise encourages exploration, but too much noise may prevent convergence.

    lr_a:         Learning rate for the Actor network.
    Higher values speed up learning but may cause instability; lower values make learning slower but more stable.

    lr_c:         Learning rate for the Critic network.
    Same as lr_a, but for the Critic; improper values can cause divergence or slow learning.

    batch_size:   Number of transition samples drawn from the replay buffer for each network update.
    Larger batches provide more stable gradients but require more memory and computation; smaller batches increase updates' variance.
    '''

	# Adjust for different environment    
    if writer is None:
        writer = SummaryWriter("./tb_record_pendulum")
    replay_size =  int(1e5)
    warm_up = int(5e3) # 25 episodes for exploration
    reward_scale = 1.0 # 100% of original reward


    hidden_size = 256 # We use [400, 300] for hidden dimensions
    updates_per_step = 1

    ewma_reward = 0
    rewards = [0]
    ewma_reward_history = [0]
    total_numsteps = 0
    updates = 0


    # Create main components
    agent = DDPG(env.observation_space.shape[0], env.action_space, gamma, tau, hidden_size, lr_a=lr_a, lr_c=lr_c)
    ounoise = OUNoise(env.action_space.shape[0])
    memory = ReplayMemory(replay_size)

    try:
        for i_episode in range(num_episodes):
            ounoise.scale = noise_scale * (1 - i_episode / num_episodes) # Noise decay
            ounoise.reset()

            total_numsteps = agent_interact(writer, env, agent, memory, ounoise, total_numsteps, warm_up, reward_scale)
            if len(memory) >= max(warm_up, batch_size):
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
        print(f"\nCould NOT Solve!!!\n")
        if save_model: agent.save_model(env_name + "_timeout", '.pth')
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
