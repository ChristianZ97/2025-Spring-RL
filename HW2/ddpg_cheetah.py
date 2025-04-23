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

env_name = 'HalfCheetah-v3'
random_seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make(env_name)

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

        self.fc1 = nn.Linear(in_features=num_inputs, out_features=hidden_size)
        # self.ln1 = nn.LayerNorm(normalized_shape=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        # self.ln2 = nn.LayerNorm(normalized_shape=hidden_size)
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=num_outputs)

        for layer in [self.fc1, self.fc2, self.fc_out]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)
        
        ########## END OF YOUR CODE ##########
        
    def forward(self, inputs):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your actor network

        d = next(self.parameters()).device
        inputs = inputs.to(d, non_blocking=True)

        x = self.fc1(inputs)
        # x = self.ln1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        # x = self.ln2(x)
        x = torch.relu(x)

        x = self.fc_out(x)
        action = torch.tanh(x)

        action_high = torch.tensor(self.action_space.high, dtype=torch.float32, device=d)
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

        self.fc1 = nn.Linear(in_features=num_inputs, out_features=hidden_size)
        self.ln1 = nn.LayerNorm(normalized_shape=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.ln2 = nn.LayerNorm(normalized_shape=hidden_size)
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=num_outputs)

        self.fc_a = nn.Linear(in_features=num_outputs, out_features=hidden_size)

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

        x = self.fc1(inputs)
        x = self.ln1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = torch.relu(x)

        a = self.fc_a(actions)

        x = torch.add(x, a)
        x = torch.relu(x)
        q_value = self.fc_out(x)
        return q_value
        
        ########## END OF YOUR CODE ##########        
        

class DDPG(object):
    def __init__(self, num_inputs, action_space, gamma=0.995, tau=0.0005, hidden_size=128, lr_a=3e-4, lr_c=1e-3):

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
        batch = Transition(*zip(*batch))
        state_batch = torch.stack(batch.state).to(d, non_blocking=True)
        action_batch = torch.stack(batch.action).to(d, non_blocking=True)
        reward_batch = torch.stack(batch.reward).unsqueeze(1).to(d, non_blocking=True)
        mask_batch = torch.stack(batch.mask).unsqueeze(1).to(d, non_blocking=True)
        next_state_batch = torch.stack(batch.next_state).to(d, non_blocking=True)

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

def train(
    env=env,
    num_episodes=500000,
    gamma=0.99,
    tau=0.005,
    noise_scale=0.3,
    lr_a=3e-4,
    lr_c=1e-3,
    updates_per_step=4,
    render=True,
    save_model=True
    ):

    torch.autograd.set_detect_anomaly(True)

    #num_episodes = 500000
    #gamma = 0.99
    #tau = 0.005
    hidden_size = 256
    #noise_scale = 0.3
    replay_size = 1000000
    batch_size = 256
    #updates_per_step = 4
    print_freq = 5
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

        hard_update(agent.actor_perturbed, agent.actor)
        agent.actor_perturbed = agent.actor_perturbed.to("cpu")
        states, actions, masks, next_states, rewards_ep = [], [], [], [], []
        while True:
            
            ########## YOUR CODE HERE (15~25 lines) ##########
            # 1. Interact with the env to get new (s,a,r,s') samples 
            # 2. Push the sample to the replay buffer
            # 3. Update the actor and the critic

            state_tensor = torch.tensor(state_np, dtype=torch.float32)
            with torch.no_grad():
                mu = agent.actor_perturbed(state_tensor).numpy()
            mu = mu + ounoise.noise()
            action_np = np.clip(mu, agent.action_space.low, agent.action_space.high)
            next_state_np, reward_np, done_np, _ = env.step(action_np)
            mask_np = 1.0 - done_np

            states.append(state_np)
            actions.append(action_np)
            masks.append(mask_np)
            next_states.append(next_state_np)
            rewards_ep.append(reward_np)

            state_np = next_state_np
            episode_reward += reward_np
            total_numsteps += 1

            if done_np: break
            # End of one interacted episode

        state_b = torch.as_tensor(np.stack(states) , dtype=torch.float32, device=device)
        action_b = torch.as_tensor(np.stack(actions) , dtype=torch.float32, device=device)
        mask_b = torch.as_tensor(np.stack(masks) , dtype=torch.float32, device=device)
        next_state_b = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device)
        reward_b = torch.as_tensor(np.stack(rewards_ep), dtype=torch.float32, device=device)

        for s, a, m, ns, r in zip(state_b, action_b, mask_b, next_state_b, reward_b):
            memory.push(s, a, m, ns, r)

        if len(memory) >= batch_size:
            for _ in range(updates_per_step):

                batch = memory.sample(batch_size)
                value_loss, policy_loss = agent.update_parameters(batch=batch)
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
                action = agent.select_action(state=state)

                # next_state, reward, done, _ = env.step(action.numpy()[0])
                action_np = action.cpu().numpy()
                next_state_np, reward_np, done_np, _ = env.step(action_np)

                if render: env.render()
                
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

            # if ewma_reward > -120 and updates > 200: SOLVED = True
            if ewma_reward > 5000 and total_numsteps > 500: SOLVED = True
            # End one testing epoch

        if SOLVED:
            if save_model: agent.save_model(env_name, '.pth')
            print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(ewma_reward, t))
            env.render()
            env.close()
            writer.close()
            break

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
    env.seed(random_seed)  
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)  
    train()
