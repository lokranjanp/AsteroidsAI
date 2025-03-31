import torch
import torch.nn as nn
import torch.optim as optim
import os
from einops import rearrange


class TransformerQNetwork(nn.Module):
    def __init__(self, state_size, action_size, num_layers=2, heads=4, dim_feedforward=256):
        super(TransformerQNetwork, self).__init__()
        self.embedding = nn.Linear(state_size, dim_feedforward)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=heads,
            dim_feedforward=dim_feedforward,
            activation="relu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(dim_feedforward, action_size)

    def forward(self, x):
        x = self.embedding(x)  # Convert state vector to embedding space
        x = rearrange(x, 'b d -> 1 b d')  # Reshape for transformer (seq_len=1)
        x = self.transformer(x)
        x = rearrange(x, '1 b d -> b d')  # Reshape back
        return self.fc_out(x)  # Output Q-values

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
        self.model.to(self.device)

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
            done = (done,)

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
