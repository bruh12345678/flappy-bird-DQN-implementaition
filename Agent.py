import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, action_size) # 2 actions: jump or not

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.9
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.0001
        self.batch_size = 256
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.criterion = nn.MSELoss()

        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval() 

        self.train_step = 0
        self.target_update_freq = 200

    def act(self, state):
        """Choose an action: Explore (random) or Exploit (model prediction)"""
        if np.random.rand() < self.epsilon:
            return 1 if np.random.rand() < 0.08 else 0  # Random action
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values).item()  # Select action with highest Q-value

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train the DQN model using mini-batches from replay buffer"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute target Q-values using Bellman Equation
        target_q_values = rewards + (1 - dones) * self.gamma * self.target_model(next_states).max(1)[0]


        # Compute predicted Q-values for taken actions
        q_values = self.model(states).gather(1, actions).squeeze()

        # Compute loss and backpropagate
        self.loss = self.criterion(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        print(self.loss)
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay


    def save(self, filename="bird_dqn.pth"):
        """Save model and memory."""
        data = {
            "model_state": self.model.state_dict(),
            "target_model_state": self.target_model.state_dict(),  # Save target model too
            "memory": list(self.memory),  # Convert deque to list
            "epsilon": self.epsilon,
        }
        torch.save(data, filename)

    def load(self, filename="bird_dqn.pth"):
        """Load model and memory."""
        try:
            data = torch.load(filename)
            self.model.load_state_dict(data["model_state"])
            self.target_model.load_state_dict(data["target_model_state"])
            self.memory = deque(data["memory"], maxlen=2000)  # Convert list back to deque
            self.epsilon = data["epsilon"]
            print(f"Loaded model from {filename}")
        except FileNotFoundError:
            print(f"No saved model found at {filename}, starting fresh.")