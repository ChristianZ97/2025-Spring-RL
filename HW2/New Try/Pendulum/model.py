# model
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

from utils import get_device

class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.device = get_device()
        self.action_high = torch.tensor(self.action_space.high).to(self.device)

        # Network Structure

        self.fc1 = nn.Linear(in_features=num_inputs, out_features=400)
        self.fc2 = nn.Linear(in_features=400, out_features=300)
        self.fc_out = nn.Linear(in_features=300, out_features=num_outputs)

        # Network Initialization

        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.fc_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc_out.bias, -3e-3, 3e-3)
        
    def forward(self, inputs):

        action_high = self.action_high

        x = self.fc1(inputs)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc_out(x)
        action = torch.tanh(x) * action_high

        return action


class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # Network Structure

        self.fc1 = nn.Linear(in_features=(num_inputs + num_outputs), out_features=400)
        self.fc2 = nn.Linear(in_features=400, out_features=300)
        self.fc_out = nn.Linear(in_features=300, out_features=1)

        # Network Initialization

        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.fc_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc_out.bias, -3e-3, 3e-3)

    def forward(self, inputs, actions):

        x = torch.cat([inputs, actions], dim=-1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        q_value = self.fc_out(x)

        return q_value
        