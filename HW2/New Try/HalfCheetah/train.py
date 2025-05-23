# train
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

def agent_interact(writer, env, agent, memory, ounoise, total_numsteps, warm_up, reward_scale):

		# Initialize state
        state_np, _ = env.reset()
        episode_actions = []
        action_low = env.action_space.low
        action_high = env.action_space.high

        # Interaction Loop
        while True:

        	# Warm Up, random sample for action space
            if total_numsteps < warm_up:
                action_np = env.action_space.sample()

            # Obtain a noised action
            else:
                state_tensor = torch.tensor(state_np, dtype=torch.float32)
                with torch.no_grad():
                    action = agent.select_action(state_tensor, ounoise)
                action_np = action.cpu().numpy()
                action_np = np.clip(action_np, action_low, action_high) # Clip only when adding noise
            
            episode_actions.append(action_np)

            # Interact with the environment
            next_state_np, reward_np, terminated, truncated, _ = env.step(action_np)
            done_np = terminated or truncated
            mask_np = 1.0 - done_np
            total_numsteps += 1

            # Update replay buffer
            reward_np = reward_np * reward_scale
            memory.push(state_np, action_np, mask_np, next_state_np, reward_np)
            state_np = next_state_np

            if total_numsteps % 1000 == 0 and len(memory) > 0:
                states = np.array([t.state for t in memory.memory], dtype=np.float32)
                actions = np.array([t.action for t in memory.memory], dtype=np.float32)

                if writer is not None:
                    writer.add_histogram('Interact/Buffer_State_Distribution', states, total_numsteps)
                    writer.add_histogram('Interact/Buffer_Action_Distribution', actions, total_numsteps)

            # Break if episode end
            if done_np: break

        episode_actions = np.array(episode_actions, dtype=np.float32)
        if writer is not None:
            writer.add_histogram('Interact/Noised_Action_Distribution', episode_actions, total_numsteps)
        return total_numsteps


def agent_update(writer, agent, memory, batch_size, total_numsteps, updates_per_step, updates):

    for _ in range(updates_per_step):

        batch = memory.sample(batch_size)
        value_loss, policy_loss, q, target_q, td_error = agent.update_parameters(batch=batch)
        updates += 1

        if writer is not None:

            writer.add_scalar('Train/Critic_Loss', value_loss, total_numsteps)
            writer.add_scalar('Train/Actor_Loss', policy_loss, total_numsteps)

            actor_grad_norm = sum(p.grad.norm() for p in agent.actor.parameters())
            critic_grad_norm = sum(p.grad.norm() for p in agent.critic.parameters())
            writer.add_scalar('Train/AC_Grad_Ratio', actor_grad_norm / critic_grad_norm, total_numsteps)

            writer.add_scalar('Train/Q_Eval', q, total_numsteps)
            writer.add_scalar('Train/Q_Target', target_q, total_numsteps)
            writer.add_scalar('Train/TD_Error', td_error, total_numsteps)

        return updates

def agent_evaluate(writer, env, agent, i_episode, rewards, ewma_reward_history, reward_scale):

    state_np, _ = env.reset()
    t, episode_reward = 0, 0
    SOLVED = False
    episode_actions = []

    while True:
        state_tensor = torch.tensor(state_np, dtype=torch.float32)
        with torch.no_grad():
            action = agent.select_action(state_tensor)
        action_np = action.cpu().numpy()

        episode_actions.append(action_np)

        next_state_np, reward_np, terminated, truncated, _ = env.step(action_np)
        done_np = terminated or truncated
        t += 1

        episode_reward += reward_np
        state_np = next_state_np
        
        if done_np: break

    episode_actions = np.array(episode_actions, dtype=np.float32)
    if writer is not None:
        writer.add_histogram('Eval/Action_Distribution', episode_actions, i_episode)

    # Update rewards and EWMA history
    rewards.append(episode_reward)
    ewma_reward = ewma_reward_history[-1]
    ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
    ewma_reward_history.append(ewma_reward)           
    print("Episode: {}, length: {}, reward: {:.2f}, ewma reward: {:.2f}".format(i_episode, t, episode_reward, ewma_reward))

    if i_episode > 20 and ewma_reward > 5000: # or -120, -150
        SOLVED = True

    if writer is not None:
        writer.add_scalar('Eval/Episode_Reward', episode_reward, i_episode)
        writer.add_scalar('Eval/EWMA_Reward', ewma_reward, i_episode)

    return SOLVED
