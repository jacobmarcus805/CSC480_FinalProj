#!/usr/bin/env python3
"""Train a Double DQN blackjack agent."""

import argparse
import csv
import os
import random
import sys

import numpy as np
import torch

from agent import BlackjackAgent
from environment import STATE_DIM, SixDeckBlackjack, legal_action_mask_batch

# --- DEFAULT HYPERPARAMETERS ---
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 1e-4
TAU = 0.005
GRAD_CLIP = 1.0
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 60000
ROLLING_WINDOW = 1000

TELEMETRY_FIELDS = [
    "episode",
    "total_reward",
    "rolling_avg_reward",
    "epsilon",
    "outcome",
    "illegal_action_count",
    "num_actions",
    "bet_amount",
    "true_count_at_bet",
    "avg_loss",
]

def parse_args():
    """Parse command-line flags."""
    parser = argparse.ArgumentParser(description="Train a blackjack DQN agent.")
    parser.add_argument("--episodes", type=int, default=200_000, help="Number of hands to train.")
    parser.add_argument("--log-interval", type=int, default=1000, help="Print a summary every N episodes.")
    parser.add_argument("--save-path", type=str, default="blackjack_card_counter_v1.pth", help="Final model path.")
    parser.add_argument(
        "--telemetry-path",
        type=str,
        default="training_telemetry.csv",
        help="CSV file for per-episode telemetry.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5000,
        help="Save a model checkpoint every N episodes (0 disables).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run greedy evaluation only (no training); requires --save-path to exist.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose mode: log every episode to stdout.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Same seed every run so results are comparable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def legal_action_mask(states: torch.Tensor) -> torch.Tensor:
    """Call the env's mask helper and get the result back on the right device."""
    # the env does this in numpy — hop to CPU, then back to wherever the tensor lives
    mask_np = legal_action_mask_batch(states.detach().cpu().numpy())
    return torch.from_numpy(mask_np).to(device=states.device)


def epsilon_for_step(steps_done: int) -> float:
    """How much random exploration is left after this many steps."""
    return EPS_END + (EPS_START - EPS_END) * np.exp(-steps_done / EPS_DECAY)


def checkpoint_path(save_path: str, episode: int) -> str:
    """Turn blackjack_card_counter_v1.pth into blackjack_card_counter_v1_ep005000.pth etc."""
    stem, ext = os.path.splitext(save_path)
    ext = ext or ".pth"
    return f"{stem}_ep{episode:06d}{ext}"


def classify_outcome(total_reward: float) -> str:
    """Did we win money, lose money, or break even?"""
    if total_reward > 0:
        return "win"
    if total_reward < 0:
        return "loss"
    return "push"


def is_illegal_step(info: dict) -> bool:
    """Did the agent try something that isn't allowed?"""
    return "Illegal" in info.get("msg", "")


def init_telemetry_csv(path: str) -> None:
    """Start a fresh telemetry CSV with column headers."""
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=TELEMETRY_FIELDS).writeheader()


def append_telemetry_row(path: str, row: dict) -> None:
    """Log one hand to the telemetry CSV."""
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=TELEMETRY_FIELDS).writerow(row)


def rolling_average(rewards: list[float], window: int = ROLLING_WINDOW) -> float:
    """Average $/hand over the last N hands."""
    if not rewards:
        return 0.0
    window_rewards = rewards[-window:]
    return float(np.mean(window_rewards))


def run_eval(args) -> None:
    """Test a saved model with --eval. No training, no CSV."""
    if not os.path.isfile(args.save_path):
        print(f"Model not found: {args.save_path}", file=sys.stderr)
        sys.exit(1)

    set_seed(args.seed)
    env = SixDeckBlackjack()
    agent = BlackjackAgent(
        state_dim=STATE_DIM,
        action_dim=7,
        batch_size=BATCH_SIZE,
        buffer_capacity=1,  # we're not training, don't need a replay buffer
    )
    agent.load(args.save_path)
    agent.policy_net.eval()

    rewards = []
    wins = losses = pushes = 0

    for episode in range(args.episodes):
        state = env.reset()
        done = False
        hand_reward = 0.0

        while not done:
            action = agent.select_action(state, env.legal_actions(), epsilon=0.0)  # no random moves
            state, reward, done, _ = env.step(action)
            hand_reward += reward

        rewards.append(hand_reward)
        outcome = classify_outcome(hand_reward)
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        else:
            pushes += 1

        if args.debug or (episode + 1) % args.log_interval == 0:
            print(
                f"Eval {episode + 1}/{args.episodes} | "
                f"hand=${hand_reward:.2f} | rolling=${rolling_average(rewards):.2f}"
            )

    print(
        f"Eval complete: mean=${np.mean(rewards):.2f}/hand | "
        f"wins={wins} losses={losses} pushes={pushes}"
    )


def train_agent(args) -> None:
    """Play hands, update the network, write telemetry, save weights."""
    set_seed(args.seed)

    env = SixDeckBlackjack()
    agent = BlackjackAgent(
        state_dim=STATE_DIM,
        action_dim=7,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        buffer_capacity=200_000,
        grad_clip=GRAD_CLIP,
    )

    init_telemetry_csv(args.telemetry_path)

    steps_done = 0
    episode_rewards: list[float] = []

    print(
        f"Training {args.episodes:,} hands | seed={args.seed} | "
        f"telemetry={args.telemetry_path}"
    )

    for episode in range(args.episodes):
        state = env.reset()
        true_count_at_bet = env._get_true_count()  # snapshot before the deal
        done = False

        total_reward = 0.0
        illegal_action_count = 0
        num_actions = 0
        episode_losses: list[float] = []
        epsilon = EPS_END

        # a single hand = bet step + however many hit/stand moves
        while not done:
            steps_done += 1  # epsilon drops per action, not per hand
            epsilon = epsilon_for_step(steps_done)
            action = agent.select_action(state, env.legal_actions(), epsilon)
            next_state, reward, done, info = env.step(action)

            num_actions += 1
            total_reward += reward
            if is_illegal_step(info):
                illegal_action_count += 1

            agent.remember(state, action, reward / 100.0, next_state, done)  # divide by 100 so rewards aren't huge
            state = next_state

            # update weights every step once we have enough memories saved up
            loss = agent.learn(legal_action_mask)
            if loss is not None:
                episode_losses.append(loss)
                agent.soft_update_target(TAU)

        bet_amount = env.current_bet
        outcome = classify_outcome(total_reward)
        avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        episode_rewards.append(total_reward)
        roll_avg = rolling_average(episode_rewards)

        append_telemetry_row(
            args.telemetry_path,
            {
                "episode": episode,
                "total_reward": round(total_reward, 2),
                "rolling_avg_reward": round(roll_avg, 2),
                "epsilon": round(epsilon, 4),
                "outcome": outcome,
                "illegal_action_count": illegal_action_count,
                "num_actions": num_actions,
                "bet_amount": bet_amount,
                "true_count_at_bet": true_count_at_bet,
                "avg_loss": round(avg_loss, 6),
            },
        )

        if args.checkpoint_interval and (episode + 1) % args.checkpoint_interval == 0:
            ckpt = checkpoint_path(args.save_path, episode + 1)  # save midway in case training dies
            agent.save(ckpt)
            print(f"Checkpoint saved: {ckpt}")

        if args.debug or episode % args.log_interval == 0:
            print(
                f"Hand {episode}/{args.episodes} | eps={epsilon:.3f} | "
                f"reward=${total_reward:.2f} | rolling=${roll_avg:.2f} | "
                f"{outcome} | illegal={illegal_action_count} | "
                f"bet=${bet_amount} | tc={true_count_at_bet} | loss={avg_loss:.4f}"
            )

    agent.save(args.save_path)
    print(f"Training complete. Model saved to {args.save_path}")
    print(f"Telemetry saved to {args.telemetry_path}")


def main():
    """Train, or run --eval if that's what was asked for."""
    args = parse_args()
    if args.eval:
        run_eval(args)
    else:
        train_agent(args)


if __name__ == "__main__":
    main()
