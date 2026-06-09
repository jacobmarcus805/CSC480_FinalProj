import random
from collections import deque
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity=100000):
        # A deque automatically pushes old memories out when it hits capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Saves a single step of experience
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Grabs a random batch of experiences to train on
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


class BlackjackAgent:
    """Double DQN agent with replay memory and masked legal-action bootstrapping."""

    def __init__(
        self,
        state_dim=6,
        action_dim=7,
        lr=1e-4,
        gamma=0.99,
        batch_size=128,
        buffer_capacity=200000,
        grad_clip=1.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.grad_clip = grad_clip

        self.policy_net = BlackjackDQN(input_dim=state_dim, output_dim=action_dim)
        self.target_net = BlackjackDQN(input_dim=state_dim, output_dim=action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(capacity=buffer_capacity)

    def select_action(self, state, legal_actions: Sequence[int], epsilon: float) -> int:
        """Epsilon-greedy action selection restricted to legal actions."""
        legal = list(legal_actions)
        if random.random() < epsilon:
            return random.choice(legal)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0).clone()

        mask = torch.full_like(q_values, float('-inf'))
        for action in legal:
            mask[action] = 0.0
        q_values = q_values + mask
        return int(q_values.argmax().item())

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.push(state, action, reward, next_state, done)

    def learn(
        self,
        legal_action_mask_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Optional[float]:
        """Run one Double DQN update if the replay buffer has enough samples."""
        if len(self.memory) <= self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_policy = self.policy_net(next_states)
            legal_next = legal_action_mask_fn(next_states)
            next_q_policy = next_q_policy.masked_fill(~legal_next, float('-inf'))
            best_next_actions = next_q_policy.argmax(1, keepdim=True)

            next_q_target = self.target_net(next_states).gather(1, best_next_actions).squeeze(1)
            next_q_target = torch.where(
                dones.bool(),
                torch.zeros_like(next_q_target),
                next_q_target,
            )
            target_q = rewards + self.gamma * next_q_target

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        return float(loss.item())

    def soft_update_target(self, tau: float = 0.005) -> None:
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1.0 - tau) * target_param.data
            )

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        try:
            state_dict = torch.load(path, weights_only=True)
        except TypeError:
            # PyTorch < 2.0 does not support weights_only.
            state_dict = torch.load(path)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


def _smoke_test() -> None:
    from environment import SixDeckBlackjack, legal_action_mask_batch

    def legal_action_mask(states: torch.Tensor) -> torch.Tensor:
        mask_np = legal_action_mask_batch(states.detach().cpu().numpy())
        return torch.from_numpy(mask_np).to(device=states.device)

    env = SixDeckBlackjack()
    agent = BlackjackAgent(batch_size=8, buffer_capacity=64)

    state = env.reset()
    legal = env.legal_actions()
    action = agent.select_action(state, legal, epsilon=0.0)
    assert action in legal, (action, legal)

    for _ in range(agent.batch_size + 1):
        state = env.reset()
        done = False
        while not done:
            legal = env.legal_actions()
            action = agent.select_action(state, legal, epsilon=1.0)
            assert action in legal
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward / 100.0, next_state, done)
            state = next_state

    loss = agent.learn(legal_action_mask)
    assert loss is not None
    agent.soft_update_target()

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        path = tmp.name
    try:
        agent.save(path)
        agent.load(path)
    finally:
        os.remove(path)

    print(f"BlackjackAgent smoke test passed (learn loss={loss:.4f})")


if __name__ == "__main__":
    _smoke_test()
