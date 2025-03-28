# Spring 2025, 535514 Reinforcement Learning
# HW1: REINFORCE with baseline and GAE

import os
import gymnasium as gym
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

# Named tuple for storing policy outputs
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
writer = SummaryWriter("./tb_record_gae")
device = torch.device("cpu")
        
class Policy(nn.Module):
    """Policy network with actor-critic architecture."""
    
    def __init__(self):
        super(Policy, self).__init__()
        
        # Get environment properties
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        
        # Network hyperparameters
        self.hidden_size = 128
        self.lambda_ = 0.95
        self.double()  # Use double precision

        # Network architecture - Improved readability with sequential modules
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_dim)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

        # Apply orthogonal initialization
        self._init_weights()
        
        # Episode history
        self.saved_actions = []
        self.rewards = []
        
        # Move to device
        self.to(device)
    
    def _init_weights(self):
        """Apply orthogonal initialization to all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(self, state):
        """Forward pass to compute action probabilities and state values."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Extract features from state
        features = self.feature_extractor(state)
        
        # Get action probabilities and state value
        action_prob = self.actor(features)
        state_value = self.critic(features)
            
        return action_prob, state_value

    def select_action(self, state):
        """Select action based on current state."""
        state = torch.FloatTensor(state).to(device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Get action probabilities and state value
        action_prob, state_value = self.forward(state)
        m = Categorical(F.softmax(action_prob, dim=-1))
        action = m.sample()

        # Store for loss calculation
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()

    def calculate_loss(self, gamma=0.999):
        """Calculate policy and value losses using GAE."""
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []

        # Extract values
        values = [saved_action.value.item() for saved_action in saved_actions]
        rewards = np.array(self.rewards)
        
        # Create done signal
        done = np.zeros_like(rewards, dtype=int)
        if len(rewards) > 0: done[-1] = 1

        # Calculate advantages and returns using GAE
        gae = GAE(gamma=gamma, lambda_=self.lambda_, num_steps=None)
        advantages, returns = gae(rewards, values, done)

        # Calculate losses
        for (log_prob_action, state_value), R, advantage in zip(saved_actions, returns, advantages):
            # Policy loss
            policy_losses.append(-log_prob_action * advantage)
            
            # Value loss
            R_tensor = torch.tensor([[R]], dtype=torch.float32, device=state_value.device)
            value_losses.append(F.mse_loss(state_value, R_tensor))

        # Combine losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        return loss

    def clear_memory(self):
        """Clear episode memory."""
        del self.rewards[:]
        del self.saved_actions[:]

class GAE:
    """Generalized Advantage Estimation."""
    
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps

    def __call__(self, rewards, values, done):
        """Calculate advantages and returns."""
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        next_advantage = 0
        next_value = values[-1]

        # Calculate advantages in reverse order
        for t in reversed(range(len(rewards))):
            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - done[t]) - values[t]
            # GAE
            advantages[t] = delta + self.gamma * self.lambda_ * next_advantage * (1 - done[t])
            next_advantage = advantages[t]
            next_value = values[t]
            
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Calculate returns
        next_return = values[-1]
        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + self.gamma * next_return * (1 - done[t])
            next_return = returns[t]

        return advantages, returns

def train(lr=3e-4, lambda_=0.95, gamma=0.99):
    """Train the policy using REINFORCE with baseline and GAE."""
    # Initialize model and optimizer
    model = Policy()
    model.lambda_ = lambda_
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = Scheduler.StepLR(optimizer, step_size=350, gamma=0.99)
  
    # Training metrics
    ewma_reward = 0
    
    # Training loop
    for i_episode in count(1):
        # Reset environment
        state, _ = env.reset()
        ep_reward = 0
        t = 0
        
        # Episode loop
        done = False
        while not done and t < 10000:
            # Select and perform action
            action = model.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            model.rewards.append(reward)
            state = next_state
            t += 1
            ep_reward += reward

        # Update policy
        loss = model.calculate_loss(gamma=gamma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.clear_memory()

        # Update metrics
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print(f'Episode {i_episode}\tlength: {t}\treward: {ep_reward:.2f}\tewma reward: {ewma_reward:.2f}')

        # Log metrics
        writer.add_scalar('Train/Episode_Reward', ep_reward, i_episode)
        writer.add_scalar('Train/Episode_Length', t, i_episode)
        writer.add_scalar('Train/EWMA_Reward', ewma_reward, i_episode)
        writer.add_scalar('Train/Loss', loss.item(), i_episode)

        # Check if solved
        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), f'./preTrained/orth_init_LunarLander_{lr}_lambda_{lambda_}.pth')
            print(f"Solved! Running reward is {ewma_reward:.2f} and the last episode ran {t} steps!")
            break


def test(name, n_episodes=10):
    """Test a trained policy."""
    model = Policy()
    model.load_state_dict(torch.load(f'./preTrained/{name}'))
    render = True
    max_episode_len = 10000
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        print(f'Episode {i_episode}\tReward: {running_reward:.2f}')
    env.close()
    

if __name__ == '__main__':
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Setup environment and parameters
    random_seed = 42
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    lambdas = [0, 0.8, 0.85, 0.88, 0.92, 0.95, 1]
    lr = 2e-4
    gamma = 0.99

    # Train and test for each lambda
    for lambda_ in lambdas:
        writer = SummaryWriter(f"./tb_record_gae/test_oth/lambda{lambda_}")
        print(f"Training with lambda = {lambda_}")
        env.reset(seed=random_seed)
        torch.manual_seed(random_seed)
        train(lr=lr, lambda_=lambda_, gamma=gamma)
        test(f'orth_init_LunarLander_{lr}_lambda_{lambda_}.pth')