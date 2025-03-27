import os
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from OU_Noise import OUActionNoise
from replay_mem import ReplayBuffer


# Directory for saving run info
RUNS_DIR = "DDPG_robot_runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=256, hidden_dim2=256, alpha=0.0001):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.alpha = alpha

        # First layer, with initial weights and biases
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.hidden_dim1)

        # Second layer, with initial weights and biases
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.hidden_dim2)

        # Output layer, with initial weights and biases
        self.mu = nn.Linear(hidden_dim2, action_dim)
        f3 = 0.003
        torch.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.tanh(self.mu(x))

        return x

    def save_checkpoint(self, filename):
        torch.save({"model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()},
               filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=256, hidden_dim2=256, alpha=0.0001):
        super(DQN, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.alpha = alpha

        # First layer, with initial weights and biases
        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim1)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.hidden_dim1)

        # Second layer, with initial weights and biases
        self.fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.hidden_dim2)

        # Part of the second layer
        self.action_value = nn.Linear(self.action_dim, self.hidden_dim2)

        # Output layer, with initial weights and biases
        self.Q = nn.Linear(self.hidden_dim2, 1)
        f3 = 0.003
        torch.nn.init.uniform_(self.Q.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.Q.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(torch.add(state_value, action_value))
        state_action_value = self.Q(state_action_value)

        return state_action_value

    def save_checkpoint(self, filename):
        torch.save({"model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()},
               filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class DDPG(nn.Module):
    def __init__(self, hyperparameter_set):
        super(DDPG, self).__init__()

        with open('Hyperparameters.yml', 'r') as f:
            all_hyperparameters_set = yaml.safe_load(f)
            self.hyperparameters = all_hyperparameters_set[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        self.state_dim = self.hyperparameters['state_dim']
        self.action_dim = self.hyperparameters['action_dim']
        self.hidden_dim1 = self.hyperparameters['hidden_dim1']
        self.hidden_dim2 = self.hyperparameters['hidden_dim2']
        self.memory_size = self.hyperparameters['memory_size']
        self.batch_size = self.hyperparameters['batch_size']
        self.gamma = self.hyperparameters['gamma']
        self.sigma = self.hyperparameters['sigma']
        self.theta = self.hyperparameters['theta']
        self.alpha = self.hyperparameters['alpha']
        self.beta = self.hyperparameters['beta']
        self.tau = self.hyperparameters['tau']

        # Q and Pi agents
        self.Actor = Actor(self.state_dim, self.action_dim, self.hidden_dim1, self.hidden_dim2, self.alpha)
        self.DQN = DQN(self.state_dim, self.action_dim, self.hidden_dim1, self.hidden_dim2, self.beta)

        # Q' and Pi' agents
        self.Actor_target = Actor(self.state_dim, self.action_dim, self.hidden_dim1, self.hidden_dim2, self.alpha)
        self.DQN_target = DQN(self.state_dim, self.action_dim, self.hidden_dim1, self.hidden_dim2, self.beta)

        self.Noise = OUActionNoise(mu=np.zeros(self.action_dim),theta=self.theta, sigma=self.sigma)

        self.memory = ReplayBuffer(self.memory_size, [self.state_dim], self.action_dim)

        self.loss_fn = nn.MSELoss()

        self.ACTOR_FILE = os.path.join(RUNS_DIR, f'DDPG_Actor.pt')
        self.ACTOR_TARGE_FILE = os.path.join(RUNS_DIR, f'DDPG_Actor_Target.pt')
        self.Q_MODEL_FILE = os.path.join(RUNS_DIR, f'DDPG_Q_model.pt')
        self.Q_TARGET_MODEL_FILE = os.path.join(RUNS_DIR, f'DDPG_Q_Target_model.pt')


    def choose_action(self, observation):
        self.Actor.eval()
        observation = torch.tensor(observation, dtype=torch.float)
        mu = self.Actor.forward(observation)
        mu_prime = mu + torch.tensor(self.Noise(),
                                 dtype=torch.float)
        self.Actor.train()
        return mu_prime.cpu().detach().numpy()


    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
                                      self.memory.sample_buffer(self.batch_size)

        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done)
        new_state = torch.tensor(new_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        state = torch.tensor(state, dtype=torch.float)

        self.Actor_target.eval()
        self.DQN_target.eval()
        self.DQN.eval()
        target_actions = self.Actor_target.forward(new_state)
        critic_value_ = self.DQN_target.forward(new_state, target_actions)
        critic_value = self.DQN.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = torch.tensor(target)
        target = target.view(self.batch_size, 1)

        self.DQN.train()
        self.DQN.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.DQN.optimizer.step()

        self.DQN.eval()
        self.Actor.optimizer.zero_grad()
        mu = self.Actor.forward(state)
        self.Actor.train()
        actor_loss = -self.DQN.forward(state, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.Actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.Actor.named_parameters()
        critic_params = self.DQN.named_parameters()
        target_actor_params = self.Actor_target.named_parameters()
        target_critic_params = self.DQN_target.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()

        self.DQN_target.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.Actor_target.load_state_dict(actor_state_dict)

        """
        #Verify that the copy assignment worked correctly
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(target_critic_params)
        actor_state_dict = dict(target_actor_params)
        print('\nActor Networks', tau)
        for name, param in self.actor.named_parameters():
            print(name, T.equal(param, actor_state_dict[name]))
        print('\nCritic Networks', tau)
        for name, param in self.critic.named_parameters():
            print(name, T.equal(param, critic_state_dict[name]))
        input()
        """


    def eval(self):
        self.Actor.eval()
        self.Actor_target.eval()
        self.DQN.eval()
        self.DQN_target.eval()


    def save_models(self):
        print('... saving models ...')
        self.Actor.save_checkpoint(self.ACTOR_FILE)
        self.DQN.save_checkpoint(self.Q_MODEL_FILE)
        self.Actor_target.save_checkpoint(self.ACTOR_TARGE_FILE)
        self.DQN_target.save_checkpoint(self.Q_TARGET_MODEL_FILE)


    def load_models(self):
        print('... loading models ...')
        self.Actor.load_checkpoint(self.ACTOR_FILE)
        self.DQN.load_checkpoint(self.Q_MODEL_FILE)
        self.Actor_target.load_checkpoint(self.ACTOR_TARGE_FILE)
        self.DQN_target.load_checkpoint(self.Q_TARGET_MODEL_FILE)
