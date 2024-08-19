import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

HIDDEN_SIZE = 512

class DQN(nn.Module):

    def __init__(self, input_size : int, output_size : int, name : str) -> None:
        super().__init__()
        self.name = name
        self.linear1 = nn.Linear(input_size, HIDDEN_SIZE)
        self.linear2 = nn.Linear(HIDDEN_SIZE, output_size)
        self.load()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self):
        file_name = self.name + '.pth'
        model_folder_path = './saved'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
    
    def load(self):
        model_folder_path = './saved'
        file_name = os.path.join(model_folder_path, self.name + '.pth')
        if not os.path.exists(model_folder_path) or not os.path.isfile(file_name): return
        self.load_state_dict(torch.load(file_name))
        

class QTrainer(): # Double DQL
    def __init__(self, model, lr, gamma) -> None:
        self.lr = lr
        self.model = model
        self.gamma = gamma
        self.optimiser = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.int64)
        dones = torch.tensor(dones, dtype=torch.int64)
        
        curr_Qs = self.model(states)
        # Get the Q values for the actions taken
        curr_Qs = curr_Qs.gather(1, actions.unsqueeze(1)).squeeze(1)
        # Calculate the Q for next state using target model
        with torch.no_grad():
            next_Qs = self.model(next_states).max(1)[0]
            target_Qs = rewards + (1 - dones) * self.gamma * next_Qs

        self.optimiser.zero_grad()
        loss = self.criterion(curr_Qs, target_Qs)
        loss.backward()
        self.optimiser.step()


class DoubleQTrainer(): # Double DQL
    def __init__(self, model : DQN, target_model : DQN, lr : float, gamma : float) -> None:
        self.lr = lr
        self.model = model
        self.target_model = target_model
        self.tau = 0.01
        self.gamma = gamma
        self.optimiser = optim.Adam(model.parameters(), lr=self.lr)
        self.mse = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.int64)
        dones = torch.tensor(dones, dtype=torch.int64)
        
        curr_Qs = self.model(states)
        # Get the Q values for the actions taken
        curr_Qs = curr_Qs.gather(1, actions.unsqueeze(1)).squeeze(1)
        # Calculate the Q for next state using target model
        with torch.no_grad():
            next_Qs = self.model(next_states).max(1)[0]
            target_Qs = rewards + (1 - dones) * self.gamma * next_Qs

        self.optimiser.zero_grad()
        loss = self.mse(curr_Qs, target_Qs)
        loss.backward()
        self.optimiser.step()



        