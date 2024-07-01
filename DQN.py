import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save_model(self, file_name='model.pth'):
        model_folder = './model'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        file_name = os.path.join(model_folder, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, alpha, gamma):
        self.learning_rate = alpha
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train_step(self, state, action, reward, next_state, done):
        state = state.clone().detach().float().to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)

        action = torch.tensor(action, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done, )

        predicted = self.model(state)
        target = predicted.clone()

        for i in range(len(done)):
            q_new = reward[i]
            if not done[i]:
                q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(predicted, target)
        loss.backward()
        self.optimizer.step()

