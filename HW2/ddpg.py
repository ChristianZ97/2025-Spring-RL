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

from torch.optim.lr_scheduler import StepLR

env_name = 'Pendulum-v0'
random_seed = 42


if torch.cuda.is_available(): device = torch.device("cuda")
elif torch.backends.mps.is_available(): device = torch.device("mps")
else: device = torch.device("cpu")
print(f"\n Using device {device}\n")

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
# writer = SummaryWriter("./tb_record_pendulum")

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
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        #self.fc3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=num_outputs)

        for layer in [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)
        nn.init.uniform_(self.fc_out.weight, -1e-4, 1e-4)
        nn.init.constant_(self.fc_out.bias, 0)
        
        ########## END OF YOUR CODE ##########
        
    def forward(self, inputs):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your actor network

        d = next(self.parameters()).device
        inputs = inputs.to(d, non_blocking=True)

        x = self.fc1(inputs)
        x = torch.relu(x)

        x = self.fc2(x)
        x = torch.relu(x)

        #x = self.fc3(x)
        #x = torch.relu(x)

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

        self.fc1 = nn.Linear(in_features=num_inputs, out_features=(hidden_size * 2))
        self.fc2 = nn.Linear(in_features=(hidden_size * 2 + num_outputs), out_features=(hidden_size * 2))
        self.fc3 = nn.Linear(in_features=(hidden_size * 2), out_features=(hidden_size * 2))
        self.fc_out = nn.Linear(in_features=(hidden_size * 2), out_features=1)

        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)
        nn.init.uniform_(self.fc_out.weight, -1e-3, 1e-3)
        nn.init.constant_(self.fc_out.bias, 0)

        ########## END OF YOUR CODE ##########

    def forward(self, inputs, actions):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your critic network

        d = next(self.parameters()).device
        inputs = inputs.to(d, non_blocking=True)
        actions = actions.to(d, non_blocking=True)

        #x = torch.cat([inputs, actions], dim=-1)
        x = self.fc1(inputs)
        x = torch.relu(x)

        x = torch.cat([x, actions], dim=-1)
        x = self.fc2(x)
        x = torch.relu(x)

        x = self.fc3(x)
        x = torch.relu(x)

        q_value = self.fc_out(x)
        q_value = torch.clamp(q_value, min=-100.0, max=100.0)
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
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c, weight_decay=5e-3)

        self.gamma = gamma
        self.tau = tau

        self.action_low = torch.tensor(self.action_space.low).to(device)
        self.action_high = torch.tensor(self.action_space.high).to(device)

        self.actor_scheduler = StepLR(self.actor_optim, step_size=100, gamma=0.5)
        self.critic_scheduler = StepLR(self.critic_optim, step_size=100, gamma=0.5)

        
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)


    def select_action(self, state, action_noise=None):

        d = next(self.actor.parameters()).device
        state = state.to(d, non_blocking=True)

        # self.actor.eval()
        # mu = self.actor((Variable(state)))
        mu = self.actor(state)
        mu = mu.data

        ########## YOUR CODE HERE (3~5 lines) ##########
        # Add noise to your action for exploration
        # Clipping might be needed

        with torch.no_grad():
            if action_noise is not None:
                ounoise = torch.tensor(action_noise.noise(), dtype=torch.float32, device=d)
                mu += ounoise 
            mu = torch.clamp(mu, self.action_low, self.action_high)
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

        batch = Transition(*zip(*batch))
        d = next(self.actor.parameters()).device
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=d)
        action_batch = torch.tensor(np.array(batch.action), dtype=torch.float32, device=d)
        reward_batch = torch.tensor(np.array(batch.reward), dtype=torch.float32, device=d).unsqueeze(1)
        mask_batch = torch.tensor(np.array(batch.mask), dtype=torch.float32, device=d).unsqueeze(1)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=d)

        reward_batch = reward_batch * 10.0
            
        ########## YOUR CODE HERE (10~20 lines) ##########
        # Calculate policy loss and value loss
        # Update the actor and the critic

        with torch.no_grad():
            next_action = self.actor_target.forward(inputs=next_state_batch)
            target_q = self.critic_target.forward(inputs=next_state_batch, actions=next_action)
            target_q = torch.clamp(target_q, min=-100.0, max=100.0)
            td_target = reward_batch + self.gamma * mask_batch * target_q

        q = self.critic.forward(inputs=state_batch, actions=action_batch)
        td_error = (td_target - q).detach()

        value_loss = F.mse_loss(input=q, target=td_target)
        self.critic_optim.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optim.step()


        action = self.actor.forward(inputs=state_batch)
        policy_loss = -(self.critic.forward(inputs=state_batch, actions=action)).mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optim.step()

        self.actor_scheduler.step()
        self.critic_scheduler.step()


        ########## END OF YOUR CODE ##########

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item(), q.mean().item(), target_q.mean().item(), td_error.abs().mean().item()


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
    num_episodes=5000,
    gamma=0.99,
    tau=0.005,
    noise_scale=0.3,
    lr_a=3e-4,
    lr_c=1e-3,
    updates_per_step=1,
    hidden_size=256,
    batch_size=64,
    render=True,
    save_model=True,
    writer=None
    ):

    torch.autograd.set_detect_anomaly(True)
    if writer is None:
        writer = SummaryWriter("./tb_record_pendulum")

    replay_size = int(5e4)
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
        
        progress = i_episode / 300
        decay = np.exp(-5 * progress)  # 從1衰減到~0.05
        ounoise.scale = max(0.05, noise_scale * decay)
        # ounoise.scale = noise_scale
        ounoise.reset()
        
        # state = torch.Tensor([env.reset()])
        state_np = env.reset()

        episode_reward = 0

        hard_update(agent.actor_perturbed, agent.actor_target)
        agent.actor_perturbed = agent.actor_perturbed.to("cpu")
        while True:
            
            ########## YOUR CODE HERE (15~25 lines) ##########
            # 1. Interact with the env to get new (s,a,r,s') samples 
            # 2. Push the sample to the replay buffer
            # 3. Update the actor and the critic

            state_tensor = torch.tensor(state_np, dtype=torch.float32)
            
            if total_numsteps < 1000:
                action_np = env.action_space.sample()
            else:
                with torch.no_grad():
                    mu = agent.actor_perturbed(state_tensor).numpy()
                mu = mu + ounoise.noise()
                action_np = np.clip(mu, agent.action_space.low, agent.action_space.high)

            next_state_np, reward_np, done_np, _ = env.step(action_np)
            total_numsteps += 1
            mask_np = 1.0 - done_np

            memory.push(state_np, action_np, mask_np, next_state_np, reward_np)

            state_np = next_state_np
            episode_reward += reward_np

            if done_np: break
        # End of one interacted episode

        if len(memory) >= 1000:
            for _ in range(updates_per_step):

                batch = memory.sample(batch_size)
                value_loss, policy_loss, q, target_q, td_error = agent.update_parameters(batch=batch)
                updates += 1

                writer.add_scalar('Update/Critic_Loss', value_loss, total_numsteps)
                writer.add_scalar('Update/Actor_Loss', policy_loss, total_numsteps)

                actor_grad_norm = sum(p.grad.norm() for p in agent.actor.parameters())
                critic_grad_norm = sum(p.grad.norm() for p in agent.critic.parameters())
                writer.add_scalar('Update/AC_Grad_Ratio', actor_grad_norm / critic_grad_norm, total_numsteps)

                writer.add_scalar('Update/Q_Eval', q, total_numsteps)
                writer.add_scalar('Update/Q_Target', target_q, total_numsteps)
                writer.add_scalar('Update/TD_Error', td_error, total_numsteps)

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
                with torch.no_grad():
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

            if i_episode > 30:
                if ewma_reward > -120: SOLVED = True
                # if ewma_reward > 5000: SOLVED = True
            # End one testing epoch

        writer.add_scalar('Train/Episode_Reward', rewards[-1], i_episode)
        writer.add_scalar('Train/EWMA_Reward', ewma_reward_history[-1], i_episode)

        if SOLVED:
            if save_model: agent.save_model(env_name, '.pth')
            print("\nSolved! Running reward is now {} and "
              "the last episode runs to {} time steps!\n".format(ewma_reward, t))
            if render: env.render()
            env.close()
            writer.close()
            break

    print("\nUnsolved! Reach the MAXIMUM num_episodes!\n")
    if save_model: agent.save_model(env_name, '.pth')
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
