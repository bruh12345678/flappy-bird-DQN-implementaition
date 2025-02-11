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
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, action_size)  # 2 actions: jump or not

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return self.fc6(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)  # Replay buffer
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1  # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.999  # Decay factor for exploration
        self.learning_rate = 0.001
        self.batch_size = 128
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())  # Copy weights
        self.target_model.eval()


    def act(self, state):
        """Choose an action: Explore (random) or Exploit (model prediction)"""
        if np.random.rand() < self.epsilon:
            return 1 if np.random.rand() < 0.07 else 0  # Random action
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values).item()  # Select action with highest Q-value

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, episode):
        """Train the DQN model using mini-batches from replay buffer"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor([float(d) for d in dones]).to(self.device)

        if episode % 10 == 0:  
            self.target_model.load_state_dict(self.model.state_dict())
        # Compute target Q-values using Bellman Equation
        target_q_values = rewards + (1 - dones) * self.gamma * torch.max(self.target_model(next_states), dim=1)[0]

        # Compute predicted Q-values for taken actions
        q_values = self.model(states).gather(1, actions).squeeze()

        # Compute loss and backpropagate
        loss = self.criterion(q_values, target_q_values.detach())

        print(f"loss for episode {episode}:{loss}")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.detach().item()


    def save(self, filename="bird_dqn.pth"):
        """Save model and memory."""
        data = {
            "model_state": self.model.state_dict(),
            "memory": self.memory,
            "epsilon": self.epsilon,
        }
        torch.save(data, filename)

    def load(self, filename="bird_dqn.pth"):
        """Load model and memory."""
        try:
            data = torch.load(filename, weights_only=False)
            self.model.load_state_dict(data["model_state"])
            self.memory = deque(data["memory"], maxlen=2000)  # Convert list back to deque
            self.epsilon = data["epsilon"]
        except FileNotFoundError:
            print(f"No saved model found at {filename}, starting fresh.")