# Spring 2025, 535514 Reinforcement Learning
# HW1: REINFORCE with baseline and GAE

import os
import gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
writer = SummaryWriter("./tb_record_gae")
device = torch.device("cpu")
        
class Policy(nn.Module):
        
    def __init__(self):
        super(Policy, self).__init__()
        
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128
        self.lambda_ = 0.95
        self.double()

        self.obs_layer1 = nn.Linear(self.observation_dim, self.hidden_size)
        self.obs_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.obs_layer3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.obs_layer4 = nn.Linear(self.hidden_size, self.hidden_size)

        self.act_layer1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_layer3 = nn.Linear(self.hidden_size, self.action_dim)

        self.val_layer1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.val_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.val_layer3 = nn.Linear(self.hidden_size, 1)

        for layer in [self.obs_layer1, self.obs_layer2, self.obs_layer3, self.obs_layer4,
                      self.act_layer1, self.act_layer2, self.act_layer3,
                      self.val_layer1, self.val_layer2, self.val_layer3]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)

        self.saved_actions = []
        self.rewards = []

        self.to(device)

    def forward(self, state):

        if state.dim() == 1:
            state = state.unsqueeze(0)

        obs = F.relu(self.obs_layer1(state))
        obs = F.relu(self.obs_layer2(obs))
        obs = F.relu(self.obs_layer3(obs))
        obs = F.relu(self.obs_layer4(obs))

        act = F.relu(self.act_layer1(obs))
        act = F.relu(self.act_layer2(act))
        action_prob = self.act_layer3(act)

        val = F.relu(self.val_layer1(obs))
        val = F.relu(self.val_layer2(val))
        state_value = self.val_layer3(val)
            
        return action_prob, state_value


    def select_action(self, state):

        state = torch.FloatTensor(state).to(device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        action_prob, state_value = self.forward(state)
        m = Categorical(F.softmax(action_prob, dim=-1))
        action = m.sample()

        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()


    def calculate_loss(self, gamma=0.999):

        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = [] 

        values = [saved_action.value.item() for saved_action in saved_actions]
        rewards = np.array(self.rewards)

        done = np.zeros_like(rewards, dtype=int)
        if len(rewards) > 0: done[-1] = 1

        gae = GAE(gamma=gamma, lambda_=self.lambda_, num_steps = None)
        advantages, returns = gae(rewards, values, done)

        for (log_prob_action, state_value), R, advantage in zip(saved_actions, returns, advantages):

            policy_losses.append(-log_prob_action * advantage)

            R_tensor = torch.tensor([R], dtype=torch.float32, device=state_value.device)
            value_losses.append(F.mse_loss(state_value, R_tensor))

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        return loss

    def clear_memory(self):
        del self.rewards[:]
        del self.saved_actions[:]

class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps

    def __call__(self, rewards, values, done):
        returns = np.zeros_like(rewards)
        next_advantage = 0
        next_value = values[-1]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - done[t]) - values[t]
            advantages[t] = delta + self.gamma * self.lambda_ * next_advantage * (1 - done[t])
            next_advantage = advantages[t]
            next_value = values[t]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        next_return = values[-1]
        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + self.gamma * next_return * (1 - done[t])
            next_return = returns[t]

        return advantages, returns

def train(lr=3e-4, lambda_=0.95):

    model = Policy()
    model.lambda_ = lambda_ 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = Scheduler.StepLR(optimizer, step_size=350, gamma=0.99)
  
    ewma_reward = 0
    
    for i_episode in count(1):
            
        state = env.reset()
        ep_reward = 0
        t = 0
        scheduler.step()
        
        done = False
        while not done and t < 10000:
            action = model.select_action(state)
            next_state, reward, done, _ = env.step(action)
            model.rewards.append(reward)
            state = next_state
            t += 1
            ep_reward += reward

        loss = model.calculate_loss(gamma=0.99)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.clear_memory()

        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        writer.add_scalar('Train/Episode_Reward', ep_reward, i_episode)
        writer.add_scalar('Train/Episode_Length', t, i_episode)
        writer.add_scalar('Train/EWMA_Reward', ewma_reward, i_episode)
        writer.add_scalar('Train/Loss', loss.item(), i_episode)

        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/orth_init_LunarLander_{}_lambda_{}.pth'.format(lr, lambda_))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


def test(name, n_episodes=10):
    model = Policy()
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    render = True
    max_episode_len = 10000
    
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    random_seed = 42
    env = gym.make('LunarLander-v2')
    lambdas = [0, 0.8, 0.85, 0.88, 0.92, 0.95, 1]
    lr = 2e-4

    for lambda_ in lambdas:
        writer = SummaryWriter(f"./tb_record_gae/test_oth/lambda{lambda_}")
        print(f"Training with lambda = {lambda_}")
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        train(lr=lr, lambda_=lambda_)
        test('./orth_init_LunarLander_{}_lambda_{}.pth'.format(lr, lambda_))
