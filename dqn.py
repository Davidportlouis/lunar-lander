import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, lr, ip_dims, fc1_dims, fc2_dims, n_actions):
        super(DQN, self).__init__()
        
        #instance variables
        self.ip_dims = ip_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        #nn layers
        self.fc1 = nn.Linear(*self.ip_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.out = nn.Linear(self.fc2_dims, self.n_actions)
        
        #optimizer
        self.optimizer = optim.Adam(self.parameters(), lr)

        #loss function
        self.criterion = nn.MSELoss()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


class Agent(object):
    def __init__(self, gamma, epsilon, lr, ip_dims, batch_size, n_actions, max_mem=100000,
            eps_end=0.01, eps_dec=5e-4):
        
        #instance variables
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem
        self.batch_size = batch_size
        self.mem_ctr = 0

        self.Q_eval = DQN(self.lr, ip_dims, 256, 256, n_actions)

        self.state_mem = np.zeros((self.mem_size, *ip_dims), dtype=np.float32)
        self.new_state_mem = np.zeros((self.mem_size, *ip_dims), dtype=np.float32)

        self.action_mem = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_mem = np.zeros(self.mem_size,dtype=np.float32)
        self.terminal_mem = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_ctr % self.mem_size
        self.state_mem[index] = state
        self.new_state_mem[index] = state_
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.terminal_mem[index] = done
        self.mem_ctr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_ctr < self.batch_size:
            return 
        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_ctr,self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = torch.tensor(self.state_mem[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_mem[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_mem[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_mem[batch]).to(self.Q_eval.device)

        action_batch = self.action_mem[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index,action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.criterion(q_target,q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
