import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from environment import SixDeckBlackjack, STATE_DIM
from agent import ReplayBuffer, BlackjackDQN

# --- HYPERPARAMETERS ---
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 1e-4    # Lower LR for stability with Huber + soft updates
EPISODES = 200000
TAU = 0.005             # Soft target update rate (Polyak)
GRAD_CLIP = 1.0

# Epsilon parameters (in env steps)
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 60000


def legal_action_mask(states: torch.Tensor) -> torch.Tensor:
    """Vectorized legal-action mask for a batch of state tensors.

    Mirrors `SixDeckBlackjack.legal_actions()` so that bootstrap targets and
    greedy action selection both ignore illegal actions. Without this, the
    `max` over Q-values during the target computation leaks the values of
    actions the agent never actually faces, biasing learning.

    Layout: state[:, 0]=phase, state[:, 5]=num_cards/10.
    """
    B = states.shape[0]
    mask = torch.zeros(B, 7, dtype=torch.bool, device=states.device)

    phase = states[:, 0]
    num_cards = (states[:, 5] * 10).round()

    is_phase0 = phase < 0.5
    is_phase1 = ~is_phase0
    has_two_cards = num_cards == 2

    # Phase 0: bets 0, 1, 2 legal
    mask[is_phase0, 0] = True
    mask[is_phase0, 1] = True
    mask[is_phase0, 2] = True

    # Phase 1: hit (3) and stand (4) always legal
    mask[is_phase1, 3] = True
    mask[is_phase1, 4] = True
    # Double (5) only legal on first two cards
    mask[is_phase1 & has_two_cards, 5] = True

    # Action 6 (split) is never legal in this build.
    return mask


def select_action(env, state, policy_net, epsilon):
    """Epsilon-greedy action selection that always respects legal_actions()."""
    legal = env.legal_actions()
    if random.random() < epsilon:
        return random.choice(legal)

    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(state_tensor).squeeze(0).clone()

    mask = torch.full_like(q_values, float('-inf'))
    for a in legal:
        mask[a] = 0.0
    q_values = q_values + mask
    return int(q_values.argmax().item())


def soft_update(target_net, policy_net, tau):
    for tp, sp in zip(target_net.parameters(), policy_net.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def train_agent():
    print("Initializing Casino and AI...")
    env = SixDeckBlackjack()
    memory = ReplayBuffer(capacity=200000)

    policy_net = BlackjackDQN(input_dim=STATE_DIM, output_dim=7)
    target_net = BlackjackDQN(input_dim=STATE_DIM, output_dim=7)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.SmoothL1Loss()  # Huber loss

    steps_done = 0
    history_rewards = []
    history_epsilon = []

    print("Starting Training Loop...")
    for episode in range(EPISODES):
        state = env.reset()
        done = False
        current_episode_reward = 0

        while not done:
            steps_done += 1
            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
            action = select_action(env, state, policy_net, epsilon)

            next_state, reward, done, _ = env.step(action)
            current_episode_reward += reward

            memory.push(state, action, reward / 100.0, next_state, done)
            state = next_state

            if len(memory) > BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

                current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    # --- Double DQN with masked legality ---
                    # Select next action with the online (policy) net so we don't
                    # propagate optimistic target-net noise; evaluate with target.
                    next_q_policy = policy_net(next_states)
                    legal_next = legal_action_mask(next_states)
                    next_q_policy = next_q_policy.masked_fill(~legal_next, float('-inf'))
                    best_next_actions = next_q_policy.argmax(1, keepdim=True)

                    next_q_target = target_net(next_states).gather(1, best_next_actions).squeeze(1)
                    # Terminal states have no future return.
                    next_q_target = torch.where(
                        dones.bool(),
                        torch.zeros_like(next_q_target),
                        next_q_target,
                    )
                    target_q = rewards + GAMMA * next_q_target

                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), GRAD_CLIP)
                optimizer.step()

                # Soft update target network every step
                soft_update(target_net, policy_net, TAU)

        history_rewards.append(current_episode_reward)
        history_epsilon.append(epsilon)

        if episode % 1000 == 0:
            window = history_rewards[-1000:] if len(history_rewards) >= 1000 else history_rewards
            recent = float(np.mean(window)) if window else 0.0
            print(f"Hand {episode}/{EPISODES} | eps={epsilon:.3f} | last1k_avg=${recent:.2f}")

    print("Training Complete!")
    torch.save(policy_net.state_dict(), "blackjack_card_counter_v1.pth")
    np.save("training_telemetry.npy", {
        "rewards": history_rewards,
        "epsilons": history_epsilon,
    })


if __name__ == "__main__":
    train_agent()
