from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks.q_network import QNetwork
from utils.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(
        self, state_dim: int, action_dim: int, buffer_size: int = 100000,
        batch_size: int = 64, gamma: float = 0.99, lr: float = 1e-3,
        tau: float = 1e-3, epsilon_start: float = 1.0, epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995, target_update_freq: int = 100
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.step_counter = 0

        # initialize q network and target network
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # define optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)

    def select_action(self, state: np.ndarray) -> int:
        # choose random action or best q value 
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return int(torch.argmax(q_values).item())

    def store_transition(self, state, action, reward, next_state, done):
        # store experience in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_step(self):
        # only train if enough samples in buffer
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # sample a batch of experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # get q values for current states and selected actions
        q_values = self.q_network(states).gather(1, actions)

        # compute target q values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)

        target_q = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # compute loss and update q network
        loss = self.criterion(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

        # periodically ipdate target network
        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())