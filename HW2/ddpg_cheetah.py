# Spring 2025, 535514 Reinforcement Learning
# HW2: DDPG

import sys
import gym
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

env_name = 'HalfCheetah-v2'
random_seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure a wandb log
# #wandb.login()
#run = wandb.init(
#    project="my-ddpg-project",  # Specify your project
#    config={                    # Track hyperparameters and metadata
#        "learning_rate": 0.01,
#    },
#)

# Define a tensorboard writer
writer = SummaryWriter("./tb_record_cheetah")

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale    

class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own actor network

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, num_outputs)

        for layer in [self.fc1, self.fc2, self.fc3, self.fc_out]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)
        
        ########## END OF YOUR CODE ##########
        
    def forward(self, inputs):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your actor network

        device = next(self.parameters()).device
        inputs = inputs.to(device)

        x = F.relu(self.ln1(self.fc1(inputs)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        action = torch.tanh(self.fc_out(x))

        action_high = torch.tensor(self.action_space.high, 
                                  dtype=action.dtype, 
                                  device=device)
        scaled_action = action * action_high
        return scaled_action

        ########## END OF YOUR CODE ##########

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own critic network

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size + num_outputs, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)


        for layer in [self.fc1, self.fc2, self.fc3, self.fc_out]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)

        ########## END OF YOUR CODE ##########

    def forward(self, inputs, actions):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your critic network

        device = next(self.parameters()).device
        inputs = inputs.to(device)
        actions = actions.to(device)

        x = F.relu(self.ln1(self.fc1(inputs)))
        x = torch.cat([x, actions], 1)
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        q_value = self.fc_out(x)
        return q_value
        
        ########## END OF YOUR CODE ##########        
        

class DDPG(object):
    def __init__(self, num_inputs, action_space, gamma=0.995, tau=0.0005, hidden_size=128, lr_a=1e-4, lr_c=1e-3):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space).to(device)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space).to(device)
        self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space).to(device)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space).to(device)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space).to(device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor) 
        hard_update(self.critic_target, self.critic)

    def select_action(self, state, action_noise=None):
        self.actor.eval()
        mu = self.actor((Variable(state)))
        mu = mu.data

        ########## YOUR CODE HERE (3~5 lines) ##########
        # Add noise to your action for exploration
        # Clipping might be needed

        if action_noise is not None:
            mu += action_noise.noise()

        mu = torch.clamp(mu, torch.tensor(self.action_space.low), torch.tensor(self.action_space.high))
        return mu

        ########## END OF YOUR CODE ##########


    def update_parameters(self, batch):
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        next_state_batch = Variable(torch.cat(batch.next_state))
        
        ########## YOUR CODE HERE (10~20 lines) ##########
        # Calculate policy loss and value loss
        # Update the actor and the critic

        self.actor_target.eval()
        self.critic_target.eval()
        target_scaled_action = self.actor_target.forward(inputs=next_state_batch).to(device)
        target_q_value = self.critic_target.forward(inputs=next_state_batch, actions=target_scaled_action).to(device)
        td_target = (reward_batch + self.gamma * mask_batch * target_q_value).to(device)
        

        self.critic.train()
        eval_q_value = self.critic.forward(inputs=state_batch, actions=action_batch).to(device)
        value_loss = F.mse_loss(input=eval_q_value, target=td_target).to(device)

        self.critic_optim.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.step()


        self.actor.train()
        eval_scaled_action = self.actor.forward(inputs=state_batch).to(device)

        self.critic.eval()
        policy_loss = -self.critic.forward(inputs=state_batch, actions=eval_scaled_action).mean().to(device)

        self.actor_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optim.step()

        ########## END OF YOUR CODE ########## 

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()


    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        local_time = time.localtime()
        timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
        if not os.path.exists('preTrained/'):
            os.makedirs('preTrained/')

        if actor_path is None:
            actor_path = "preTrained/ddpg_actor_{}_{}_{}".format(env_name, timestamp, suffix) 
        if critic_path is None:
            critic_path = "preTrained/ddpg_critic_{}_{}_{}".format(env_name, timestamp, suffix) 
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path))

def train(env, num_episodes=500000, gamma=0.99, tau=0.005, noise_scale=0.2, 
          lr_a=3e-5, lr_c=3e-4, render=False, save_model=True):

    hidden_size = 512
    replay_size = 1000000
    batch_size = 256
    updates_per_step = 4
    print_freq = 5
    ewma_reward = 0
    rewards = []
    ewma_reward_history = []
    total_numsteps = 0
    updates = 0

    SOLVED = False
    
    agent = DDPG(
        num_inputs=env.observation_space.shape[0], 
        action_space=env.action_space, 
        gamma=gamma, 
        tau=tau, 
        hidden_size=hidden_size, 
        lr_a=lr_a, 
        lr_c=lr_c
        )
    ounoise = OUNoise(action_dimension=env.action_space.shape[0])
    memory = ReplayMemory(replay_size)

    policy_loss = torch.tensor(0.0).to(device)
    value_loss = torch.tensor(0.0).to(device)
    
    for i_episode in range(num_episodes):
        
        ounoise.scale = noise_scale
        ounoise.reset()
        
        # state = torch.Tensor([env.reset()])
        state = torch.from_numpy(env.reset()).float().unsqueeze(0).to(device)

        episode_reward = 0
        while True:
            
            ########## YOUR CODE HERE (15~25 lines) ##########
            # 1. Interact with the env to get new (s,a,r,s') samples 
            # 2. Push the sample to the replay buffer
            # 3. Update the actor and the critic

            buffer_states, buffer_actions = [], []
            buffer_rewards, buffer_masks, buffer_next_states = [], [], []

            with torch.no_grad():
                for _ in range(batch_size):
                    action = agent.select_action(state, action_noise=ounoise)
                    next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
                    buffer_states.append(state.cpu().numpy()[0])
                    buffer_actions.append(action.cpu().numpy()[0])
                    buffer_rewards.append(reward)
                    buffer_masks.append(0.0 if done else 1.0)
                    buffer_next_states.append(next_state)
                    state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
                    if done:
                        state = torch.from_numpy(env.reset()).float().unsqueeze(0).to(device)

            states = torch.from_numpy(np.vstack(buffer_states)).float().to(device)
            actions = torch.from_numpy(np.vstack(buffer_actions)).float().to(device)
            rewards = torch.from_numpy(np.array(buffer_rewards)).unsqueeze(1).float().to(device)
            masks = torch.from_numpy(np.array(buffer_masks)).unsqueeze(1).float().to(device)
            next_states = torch.from_numpy(np.vstack(buffer_next_states)).float().to(device)

            for i in range(batch_size):
                memory.push(
                    states[i].unsqueeze(0),
                    actions[i].unsqueeze(0),
                    masks[i].unsqueeze(0),
                    next_states[i].unsqueeze(0),
                    rewards[i].unsqueeze(0)
                )

            if len(memory) >= batch_size:
                for _ in range(updates_per_step):
                    transitions = memory.sample(batch_size=batch_size)
                    batch = Transition(*zip(*transitions))
                    value_loss, policy_loss = agent.update_parameters(batch=batch)
                    updates += 1

            ########## END OF YOUR CODE ########## 
            # For wandb logging
            # wandb.log({"actor_loss": actor_loss, "critic_loss": critic_loss})

        rewards.append(episode_reward)
        t = 0
        if i_episode % print_freq == 0:
            # state = torch.Tensor([env.reset()])
            state = torch.from_numpy(env.reset()).float().unsqueeze(0).to(device)
            episode_reward = 0
            while True:
                action = agent.select_action(state)

                next_state, reward, done, _ = env.step(action.numpy()[0])
                
                env.render()
                
                episode_reward += reward

                # next_state = torch.Tensor([next_state])
                next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)

                state = next_state
                
                t += 1
                if done:
                    break

            rewards.append(episode_reward)
            # update EWMA reward and log the results
            ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
            ewma_reward_history.append(ewma_reward)           
            print("Episode: {}, length: {}, reward: {:.2f}, ewma reward: {:.2f}".format(i_episode, t, rewards[-1], ewma_reward))
            
            writer.add_scalar('Train/EWMA_Reward', ewma_reward, i_episode)
            writer.add_scalar('Train/Episode_Length', t, i_episode)
            writer.add_scalar('Train/Actor_Loss', policy_loss, i_episode)
            writer.add_scalar('Train/Critic_Loss', value_loss, i_episode)

            # if ewma_reward > -200 and i_episode > 200: SOLVED = True
            if ewma_reward > 5000 and i_episode > 500: SOLVED = True

            if SOLVED:
                if save_model: agent.save_model(env_name, '.pth')
                print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
                env.close()
                writer.close()
                break

    if not SOLVED:
        if save_model: agent.save_model(env_name, '.pth')
        print("Unsolved! Reach the MAXIMUM num_episodes!")
        env.close()
        writer.close()

    return {
        'last_rewards': rewards[-10:],
        'best_reward': max(rewards),
        'ewma_reward': ewma_reward,
        'rewards': rewards
    }


if __name__ == '__main__':

    # For reproducibility, fix the random seed
    # env_name = 'Pendulum-v0'
    # random_seed = 42
    env = gym.make(env_name)
    env.seed(random_seed)  
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)  
    train(env)
