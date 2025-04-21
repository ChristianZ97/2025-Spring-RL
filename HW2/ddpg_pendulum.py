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

env_name = 'Pendulum-v0'
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
writer = SummaryWriter("./tb_record_pendulum")

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

        self.fc1 = nn.Linear(in_features=num_inputs, out_features=400)
        self.ln1 = nn.LayerNorm(normalized_shape=400)
        self.fc2 = nn.Linear(in_features=400, out_features=300)
        self.ln2 = nn.LayerNorm(normalized_shape=300)
        self.fc_out = nn.Linear(in_features=300, out_features=num_outputs)

        for layer in [self.fc1, self.fc2, self.fc_out]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)
        
        ########## END OF YOUR CODE ##########
        
    def forward(self, inputs):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your actor network

        d = next(self.parameters()).device
        inputs = inputs.to(d, non_blocking=True)

        x = torch.relu(self.ln1(self.fc1(inputs)))
        x = torch.relu(self.ln2(self.fc2(x)))
        action = torch.tanh(self.fc_out(x))

        action_high = torch.tensor(self.action_space.high, dtype=torch.float32, device=device)
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

        self.fc1 = nn.Linear(in_features=num_inputs, out_features=400)
        self.ln1 = nn.LayerNorm(normalized_shape=400)
        self.fc2 = nn.Linear(in_features=400, out_features=300)
        self.ln2 = nn.LayerNorm(normalized_shape=300)
        self.fc_out = nn.Linear(in_features=300, out_features=num_outputs)

        self.fc_a = nn.Linear(in_features=num_outputs, out_features=300)

        for layer in [self.fc1, self.fc2, self.fc_out, self.fc_a]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)

        ########## END OF YOUR CODE ##########

    def forward(self, inputs, actions):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your critic network

        d = next(self.parameters()).device
        inputs = inputs.to(d, non_blocking=True)
        actions = actions.to(d, non_blocking=True)

        x = torch.relu(self.ln1(self.fc1(inputs)))
        x = torch.relu(self.ln2(self.fc2(x)))

        a = self.fc_a(actions)
        q_value = self.fc_out(torch.relu(torch.add(x, a)))
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
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c, weight_decay=1e-2)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    @torch.no_grad()
    def select_action(self, state, action_noise=None):

        d = next(self.actor.parameters()).device
        state = state.to(d, non_blocking=True)

        self.actor.eval()
        # mu = self.actor((Variable(state)))
        mu = self.actor(state)
        mu = mu.data

        ########## YOUR CODE HERE (3~5 lines) ##########
        # Add noise to your action for exploration
        # Clipping might be needed

        # dtype=torch, device=gpu
        if action_noise is not None:
            ounoise = torch.tensor(action_noise.noise(), dtype=torch.float32, device=d)
            mu += ounoise
        
        action_low = torch.tensor(self.action_space.low, dtype=torch.float32, device=d)
        action_high = torch.tensor(self.action_space.high, dtype=torch.float32, device=d)
        mu = torch.clamp(mu, action_low, action_high)

        self.actor.train()
        return mu

        ########## END OF YOUR CODE ##########


    def update_parameters(self, batch):
        '''
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        next_state_batch = Variable(torch.cat(batch.next_state))
        '''
        d = next(self.actor.parameters()).device
        state_batch = batch.state.to(d, non_blocking=True)
        action_batch = batch.action.to(d, non_blocking=True)
        reward_batch = batch.reward.to(d, non_blocking=True)
        mask_batch = batch.mask.to(d, non_blocking=True)
        next_state_batch = batch.next_state.to(d, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            
            ########## YOUR CODE HERE (10~20 lines) ##########
            # Calculate policy loss and value loss
            # Update the actor and the critic

            # dtype=torch, device=gpu
            self.actor_target.eval()
            self.critic_target.eval()
            with torch.no_grad():
                target_scaled_action = self.actor_target.forward(inputs=next_state_batch)
                target_q_value = self.critic_target.forward(inputs=next_state_batch, actions=target_scaled_action)
            td_target = reward_batch + self.gamma * mask_batch * target_q_value

            self.critic.train()
            eval_q_value = self.critic.forward(inputs=state_batch, actions=action_batch)
            value_loss = F.mse_loss(input=eval_q_value, target=td_target)

            self.critic_optim.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optim.step()

            self.actor.train()
            eval_scaled_action = self.actor.forward(inputs=state_batch)

            self.critic.eval()
            policy_loss = -self.critic.forward(inputs=state_batch, actions=eval_scaled_action).mean()

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

def train():
    torch.autograd.set_detect_anomaly(True)

    num_episodes = 500000
    gamma = 0.99
    tau = 0.001
    hidden_size = 128
    noise_scale = 0.3
    replay_size = 100000
    batch_size = 256
    updates_per_step = 1
    print_freq = 1
    ewma_reward = 0
    rewards = []
    ewma_reward_history = []
    total_numsteps = 0
    updates = 0

    SOLVED = False
    
    agent = DDPG(env.observation_space.shape[0], env.action_space, gamma, tau, hidden_size)
    ounoise = OUNoise(env.action_space.shape[0])
    memory = ReplayMemory(replay_size)
    
    for i_episode in range(num_episodes):
        
        # ounoise.scale = noise_scale
        ounoise.scale = noise_scale * (1.0 - i_episode / num_episodes)
        ounoise.reset()
        
        # state = torch.Tensor([env.reset()])
        state_np = env.reset()

        episode_reward = 0
        episode_actor_loss, episode_critic_loss = 0, 0
        while True:
            
            ########## YOUR CODE HERE (15~25 lines) ##########
            # 1. Interact with the env to get new (s,a,r,s') samples 
            # 2. Push the sample to the replay buffer
            # 3. Update the actor and the critic

            # dtype=tensor, device=gpu
            state = torch.tensor(state_np, dtype=torch.float32, device=device)
            action = agent.select_action(state=state, action_noise=ounoise)

            # dtype=numpy, device=cpu
            action_np = action.cpu().numpy()
            next_state_np, reward_np, done_np, _ = env.step(action_np)
            mask_np = 1.0 - done_np
            memory.push(state_np, action_np, mask_np, next_state_np, reward_np)

            state_np = next_state_np
            episode_reward += reward_np

            if len(memory) >= batch_size:
                for _ in range(updates_per_step):

                    # dtype=numpy, device=cpu
                    transition = memory.sample(batch_size=batch_size)
                    batch_b = Transition(*zip(*transition))

                    state_b = torch.tensor(np.array(batch_b.state), dtype=torch.float32, device=device)
                    action_b = torch.tensor(np.array(batch_b.action), dtype=torch.float32, device=device)
                    mask_b = torch.tensor(np.array(batch_b.mask), dtype=torch.float32, device=device).unsqueeze(1)
                    next_state_b = torch.tensor(np.array(batch_b.next_state), dtype=torch.float32, device=device)
                    reward_b = torch.tensor(np.array(batch_b.reward), dtype=torch.float32, device=device).unsqueeze(1)

                    batch = Transition(state_b, action_b, mask_b, next_state_b, reward_b)

                    # dtype=tensor, device=gpu
                    value_loss, policy_loss = agent.update_parameters(batch=batch)
                    episode_critic_loss += value_loss
                    episode_actor_loss  += policy_loss
                    updates += 1

                    writer.add_scalar('Update/Critic_Loss', value_loss, updates)
                    writer.add_scalar('Update/Actor_Loss', policy_loss, updates)

                    with torch.no_grad():
                        q_eval = agent.critic(state_b, action_b).mean().item()
                        q_target = agent.critic_target(state_b, action_b).mean().item()
                        td_error = (q_eval - q_target).__abs__()
                    writer.add_scalar('Update/Q_Eval', q_eval, updates)
                    writer.add_scalar('Update/Q_Target', q_target, updates)
                    writer.add_scalar('Update/TD_Error', td_error, updates)

            if done_np: break
        # End one training epoch

            ########## END OF YOUR CODE ########## 
            # For wandb logging
            # wandb.log({"actor_loss": actor_loss, "critic_loss": critic_loss})

        rewards.append(episode_reward)
        t = 0
        if i_episode % print_freq == 0:
            # state = torch.Tensor([env.reset()])
            state_np = env.reset()
            episode_reward = 0
            while True:
                # action = agent.select_action(state)
                state = torch.tensor(state_np, dtype=torch.float32, device=device)
                action = agent.select_action(state=state, action_noise=ounoise)

                # next_state, reward, done, _ = env.step(action.numpy()[0])
                action_np = action.cpu().numpy()
                next_state_np, reward_np, done_np, _ = env.step(action_np)

                # env.render()
                
                # episode_reward += reward
                episode_reward += reward_np

                # next_state = torch.Tensor([next_state])

                # state = next_state
                state_np = next_state_np
                
                t += 1
                if done_np: break

            rewards.append(episode_reward)
            # update EWMA reward and log the results
            ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
            ewma_reward_history.append(ewma_reward)           
            print("Episode: {}, length: {}, reward: {:.2f}, ewma reward: {:.2f}".format(i_episode, t, rewards[-1], ewma_reward))

            writer.add_scalar('Train/Episode_Reward', rewards[-1], i_episode)
            writer.add_scalar('Train/EWMA_Reward', ewma_reward, i_episode)
            writer.add_scalar('Train/Actor_Loss', episode_actor_loss, i_episode)
            writer.add_scalar('Train/Critic_Loss', episode_critic_loss, i_episode)

            if ewma_reward > -120 and i_episode > 200: SOLVED = True
            # if ewma_reward > 5000 and i_episode > 500: SOLVED = True
        # End one testing epoch

        if SOLVED:
            if save_model: agent.save_model(env_name, '.pth')
            print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(ewma_reward, t))
            env.close()
            writer.close()
            break

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
    train()
