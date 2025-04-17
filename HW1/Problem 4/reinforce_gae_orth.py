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

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Define a tensorboard writer
writer = SummaryWriter("./tb_record_gae")

device = torch.device("cpu")
        
class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the GAE parameters, shared layer(s), the action layer(s), and the value layer(s))
            2. Random weight initialization of each layer
    """
    def __init__(self):
        super(Policy, self).__init__()
        
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128
        self.lambda_ = 0.95
        self.double()
        
        ########## YOUR CODE HERE (5~10 lines) ##########

        # Shared observation layers
        self.obs_layer1 = nn.Linear(self.observation_dim, self.hidden_size)
        self.obs_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.obs_layer3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.obs_layer4 = nn.Linear(self.hidden_size, self.hidden_size)

        # Actor head
        self.act_layer1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_layer3 = nn.Linear(self.hidden_size, self.action_dim)

        # Critic head
        self.val_layer1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.val_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.val_layer3 = nn.Linear(self.hidden_size, 1)

        for layer in [self.obs_layer1, self.obs_layer2, self.obs_layer3, self.obs_layer4,
                      self.act_layer1, self.act_layer2, self.act_layer3,
                      self.val_layer1, self.val_layer2, self.val_layer3]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)

        ########## END OF YOUR CODE ##########
        
        # action & reward memory
        self.saved_actions = []
        self.rewards = []

        self.to(device)

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding 
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########

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

        ########## END OF YOUR CODE ##########

        return action_prob, state_value


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########

        state = torch.FloatTensor(state).to(device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        action_prob, state_value = self.forward(state)
        m = Categorical(F.softmax(action_prob, dim=-1))
        action = m.sample()

        ########## END OF YOUR CODE ##########
        
        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()


    def calculate_loss(self, gamma=0.999):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
        """
        
        # Initialize the lists and variables
        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = [] 

        ########## YOUR CODE HERE (8-15 lines) ##########

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

        ########## END OF YOUR CODE ##########
        
        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps          # set num_steps = None to adapt full batch

    def __call__(self, rewards, values, done):
        """
            Implement Generalized Advantage Estimation (GAE) for your value prediction
            TODO (1): Pass correct corresponding inputs (rewards, values, and done) into the function arguments
            TODO (2): Calculate the Generalized Advantage Estimation and return the obtained value
        """

        ########## YOUR CODE HERE (8-15 lines) ##########

        advantages = np.zeros_like(rewards)
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
        
        ########## END OF YOUR CODE ##########

def train(lr=3e-4, lambda_=0.95):
    """
        Train the model using SGD (via backpropagation)
        TODO (1): In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode

        TODO (2): In each episode, 
        1. record all the value you aim to visualize on tensorboard (lr, reward, length, ...)
    """
    
    # Instantiate the policy model and the optimizer
    model = Policy()
    model.lambda_ = lambda_ 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=350, gamma=0.99)
    # scheduler = Scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200)
    
    # EWMA reward for tracking the learning progress
    ewma_reward = 0
    
    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0

        # Uncomment the following line to use learning rate scheduler
        scheduler.step()
        
        # For each episode, only run 9999 steps to avoid entering infinite loop during the learning process
        
        ########## YOUR CODE HERE (10-15 lines) ##########

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
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        optimizer.zero_grad()
        model.clear_memory()

        ########## END OF YOUR CODE ##########
            
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        #Try to use Tensorboard to record the behavior of your implementation 
        ########## YOUR CODE HERE (4-5 lines) ##########

        writer.add_scalar('Train/Episode_Reward', ep_reward, i_episode)
        writer.add_scalar('Train/Episode_Length', t, i_episode)
        writer.add_scalar('Train/EWMA_Reward', ewma_reward, i_episode)
        writer.add_scalar('Train/Loss', loss.item(), i_episode)

        ########## END OF YOUR CODE ##########

        # check if we have "solved" the cart pole problem, use 120 as the threshold in LunarLander-v2
        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/orth_init_LunarLander_{}_lambda_{}.pth'.format(lr, lambda_))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


def test(name, n_episodes=10):
    """
        Test the learned model (no change needed)
    """     
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

    # For reproducibility, fix the random seed
    random_seed = 42
    env = gym.make('LunarLander-v2')
    lambdas = [0.80, 0.81, 0.84]
    lr = 2e-4

    for lambda_ in lambdas:
        writer = SummaryWriter(f"./tb_record_gae/test_oth/lambda{lambda_}")
        print(f"Training with lambda = {lambda_}")
        # env.seed(random_seed)
        env.reset(seed=random_seed)
        torch.manual_seed(random_seed)
        train(lr=lr, lambda_=lambda_)
        test('./orth_init_LunarLander_{}_lambda_{}.pth'.format(lr, lambda_))
