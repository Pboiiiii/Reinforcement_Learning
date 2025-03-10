import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.distributions.categorical import Categorical

# Directory for saving run info
RUNS_DIR = "PPO_runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist

    def save_checkpoint(self, filename):
        T.save({"model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()},
               filename)

    def load_checkpoint(self, filename):
        checkpoint = T.load(filename)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self, filename):
        T.save({"model_state_dict":self.state_dict(),
               "optimizer_state_dict":self.optimizer.state_dict()},
               filename)

    def load_checkpoint(self, filename):
        checkpoint = T.load(filename)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class Agent:
    def __init__(self, hyperparameter_set):

        with open('Hyperparameters.yml', 'r') as f:
            all_hyperparameters_set = yaml.safe_load(f)
            self.hyperparameters = all_hyperparameters_set[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        self.gamma = self.hyperparameters['gamma']
        self.alpha = self.hyperparameters['alpha']
        self.n_epochs = self.hyperparameters['n_epochs']
        self.gae_lambda = self.hyperparameters['gae_lambda']
        self.batch_size = self.hyperparameters['batch_size']
        self.policy_clip = self.hyperparameters['policy_clip']

        self.n_actions = self.hyperparameters['n_actions']
        self.input_dims = self.hyperparameters['input_dims']

        self.actor = ActorNetwork(self.n_actions, self.input_dims, self.alpha)
        self.critic = CriticNetwork(self.input_dims, self.alpha)
        self.memory = PPOMemory(self.batch_size)

        self.ACTOR_FILE = os.path.join(RUNS_DIR, f'PPO_Actor.pt')
        self.CRITIC_FILE = os.path.join(RUNS_DIR, f'PPO_Critic.pt')

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint(self.ACTOR_FILE)
        self.critic.save_checkpoint(self.CRITIC_FILE)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint(self.ACTOR_FILE)
        self.critic.load_checkpoint(self.CRITIC_FILE)

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
                reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage)

            values = T.tensor(values)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float)
                old_probs = T.tensor(old_prob_arr[batch])
                actions = T.tensor(action_arr[batch])

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

