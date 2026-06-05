import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, capacity=100000):
        # A deque automatically pushes old memories out when it hits capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Saves a single step of experience
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Grabs a random batch of experiences to train on
        import numpy as np
        batch = random.sample(self.memory, batch_size)

        # Unpack the batch into separate PyTorch tensors
        states, actions, rewards, next_states, dones = zip(*batch)

        # Stack the per-step numpy arrays into a single ndarray first; passing a
        # list of ndarrays directly to torch.tensor is slow and noisy.
        return (
            torch.from_numpy(np.stack(states)).float(),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.from_numpy(np.stack(next_states)).float(),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.memory)

class BlackjackDQN(nn.Module):
    def __init__(self, input_dim=6, output_dim=7):
        super(BlackjackDQN, self).__init__()

        # Layer 1: Takes the state vector and expands it
        self.fc1 = nn.Linear(input_dim, 128)
        
        # Layer 2: Hidden processing layer
        self.fc2 = nn.Linear(128, 128)
        
        # Layer 3: Compresses back down to the 7 possible actions
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        """Pushes the state data through the network."""
        # Use ReLU (Rectified Linear Unit) activation functions for the hidden layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # The output layer does NOT use an activation function because we want raw Q-values
        return self.fc3(x)