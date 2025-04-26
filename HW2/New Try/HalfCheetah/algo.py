# algo
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

from model import Actor, Critic
from utils import Transition, get_device, hard_update, soft_update


class DDPG(object):
    def __init__(self, num_inputs, action_space, gamma=0.995, tau=0.0005, hidden_size=128, lr_a=1e-4, lr_c=1e-3):
        
        self.device = get_device()
        self.action_space = action_space

        # Constant
        self.num_inputs = num_inputs
        self.gamma = gamma
        self.tau = tau
        self.action_low = torch.tensor(self.action_space.low).to(self.device)
        self.action_high = torch.tensor(self.action_space.high).to(self.device)

    	# Models
        self.actor = Actor(hidden_size, self.num_inputs, self.action_space).to(self.device)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space).to(self.device)
        self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space).to(self.device)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c, weight_decay=5e-4)

        # Network Initialization
        hard_update(self.actor_target, self.actor) 
        hard_update(self.critic_target, self.critic)

    def select_action(self, state, action_noise=None):

        state = state.to(self.device)
        mu = self.actor(state)
        if action_noise:
            mu += torch.tensor(action_noise.noise(), dtype=torch.float32).to(self.device)
        return mu

    def update_parameters(self, batch):

        # Batch Preprocessing, get (s, a, r, s')
        batch = Transition(*zip(*batch))
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(np.array(batch.action), dtype=torch.float32).to(self.device)
        reward_batch = torch.tensor(np.array(batch.reward), dtype=torch.float32).unsqueeze(1).to(self.device)
        mask_batch = torch.tensor(np.array(batch.mask), dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(self.device)

        with torch.no_grad(): # Fix actor_target and critic_target
        	# target_q = Q_target(s', pi_target(s'))
            next_mu = self.actor_target.forward(next_state_batch)
            target_q = self.critic_target.forward(next_state_batch, next_mu)
            # TD Target = r + gamma * (1 - done) * Q_target(s', pi(s'))
            td_target = reward_batch + self.gamma * mask_batch * target_q

        # q = Q(s, a)
        q = self.critic.forward(state_batch, action_batch)
        # TD Error = TD Target - Q(s, a)
        td_error = (td_target - q).detach()

        # Minimize TD Error
        value_loss = F.mse_loss(q, td_target)
        self.critic_optim.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optim.step()


        # Maximize Q(s, pi(s))
        mu = self.select_action(state_batch)
        policy_loss = -(self.critic.forward(state_batch, mu)).mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optim.step()

        # Update Target Networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        # Results Postprocessing
        value_loss = value_loss.item()
        policy_loss = policy_loss.item()
        q = q.mean().item()
        target_q = target_q.mean().item()
        td_error = td_error.abs().mean().item()

        return value_loss, policy_loss, q, target_q, td_error

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

