from collections import deque
import random

class ReplayBuffer():
    def __init__(self, buffer_size, seed=1234):
        self.buffer_size = buffer_size
        self.num_experience = 0
        self.buffer = deque()
        random.seed(seed)

    def add(self, state, action, reward, state_new, done):
        experience = (state, action, reward, state_new, done)
        if self.num_experience < self.buffer_size:
            self.buffer.append(experience)
            self.num_experience += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        return self.num_experience

    def size(self):
        return self.buffer_size

    def clear(self):
        self.buffer.clear()
        self.num_experience = 0

    def sample_batch(self, batch_size):
        if self.num_experience < batch_size:
            return random.sample(self.buffer, self.num_experience)
        else:
            return random.sample(self.buffer, batch_size)



# https: // github.com / yanpanlau / DDPG - Keras - Torcs / blob / master / ReplayBuffer.py
# D:\DeepInvest\Code\DeepInv\drl_pm\src
