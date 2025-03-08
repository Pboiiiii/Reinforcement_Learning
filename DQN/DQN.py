import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=False):
        super(DQN, self).__init__()

        self.enable_dueling_dqn=enable_dueling_dqn

        self.fc1 = nn.Linear(state_dim, hidden_dim)

        if self.enable_dueling_dqn:
            # Value stream
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)

            # Advantages stream
            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim)

        else:
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        Q = self.output(x)

        return Q

    @staticmethod
    def float_to_binary(value):
        return format(np.frombuffer(np.float32(value).tobytes(), dtype=np.uint32)[0], '032b')

    @staticmethod
    def binary_to_float(binary_str):
        int_value = np.uint32(int(binary_str, 2))
        return np.frombuffer(int_value.tobytes(), dtype=np.float32)[0]


if __name__ == '__main__':
    net = DQN(5, 3, hidden_dim=10)  # Match hidden_dim with your initialization

    for name, param in net.named_parameters():
        print(f"{name}: Shape {param.shape}")

