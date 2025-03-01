#!/usr/bin/env python # -*- coding: utf-8 -*-

# @Time : 13.12.2022 14:53
# @Author : roman.held
# @Project: Deep_Q_Network.ipynb
# @File : cartpole_agent.py

import numpy as np
import random
from collections import namedtuple, deque

from cartpole_net import CartPoleNet
from experience_buffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

default_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CartPoleAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, args, state_size, action_size, device=default_device):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        if args.seed > 0:
            random.seed(args.seed)
            torch.manual_seed(args.seed)

        # Q-Network
        self.qnetwork_local = CartPoleNet(state_size, action_size, args.seed).to(self.device)
        self.qnetwork_target = CartPoleNet(state_size, action_size, args.seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=args.learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(action_size, args.replay_buffer_size, args.batch_size, args.seed, self.device)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        self.args = args

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.args.update_target_net
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > max(self.args.batch_size, self.args.replay_buffer_coldrun):
                experiences = self.memory.sample()
                return self.learn(experiences, self.args.gamma)
        return None

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.args.soft_update_tau)
        return loss

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
