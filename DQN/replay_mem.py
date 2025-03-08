# Define memory for Experience Replay
from collections import deque
import random
class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)
        self.maxlen = maxlen

        # Optional seed for reproducibility
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

    def state_dict(self):
        return {"memory": list(self.memory)}

    def load_state_dict(self, state_dict):
        self.memory = deque(state_dict, maxlen=self.maxlen)