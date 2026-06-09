#!/usr/bin/env python3
"""Run a random legal-action baseline agent."""

import argparse
import csv
import random

import numpy as np

from environment import SixDeckBlackjack

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
    parser = argparse.ArgumentParser(description="Random blackjack baseline.")
    parser.add_argument("--episodes", type=int, default=10_000, help="Number of hands to play.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--telemetry-path",
        type=str,
        default="baseline_telemetry.csv",
        help="CSV file for per-episode telemetry.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def classify_outcome(total_reward: float) -> str:
    if total_reward > 0:
        return "win"
    if total_reward < 0:
        return "loss"
    return "push"


def init_telemetry_csv(path: str) -> None:
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=TELEMETRY_FIELDS).writeheader()


def append_telemetry_row(path: str, row: dict) -> None:
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=TELEMETRY_FIELDS).writerow(row)


def rolling_average(rewards: list[float], window: int = ROLLING_WINDOW) -> float:
    if not rewards:
        return 0.0
    return float(np.mean(rewards[-window:]))


def run_baseline(args) -> None:
    set_seed(args.seed)
    env = SixDeckBlackjack()
    init_telemetry_csv(args.telemetry_path)

    episode_rewards: list[float] = []
    wins = losses = pushes = 0
    total_bet = 0
    total_actions = 0
    total_illegal = 0

    print(f"Baseline random agent | {args.episodes:,} hands | seed={args.seed}")

    for episode in range(args.episodes):
        state = env.reset()
        true_count_at_bet = env._get_true_count()
        done = False

        total_reward = 0.0
        num_actions = 0

        while not done:
            action = random.choice(env.legal_actions())
            state, reward, done, _ = env.step(action)
            total_reward += reward
            num_actions += 1

        bet_amount = env.current_bet
        outcome = classify_outcome(total_reward)
        episode_rewards.append(total_reward)
        roll_avg = rolling_average(episode_rewards)

        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        else:
            pushes += 1

        total_bet += bet_amount
        total_actions += num_actions

        append_telemetry_row(
            args.telemetry_path,
            {
                "episode": episode,
                "total_reward": round(total_reward, 2),
                "rolling_avg_reward": round(roll_avg, 2),
                "epsilon": 1.0,
                "outcome": outcome,
                "illegal_action_count": 0,
                "num_actions": num_actions,
                "bet_amount": bet_amount,
                "true_count_at_bet": true_count_at_bet,
                "avg_loss": 0.0,
            },
        )

    n = args.episodes
    bankroll = float(np.sum(episode_rewards))
    print(f"Average reward per hand: ${np.mean(episode_rewards):.2f}")
    print(f"Total bankroll: ${bankroll:.2f}")
    print(f"Win/loss/push: {wins}/{losses}/{pushes} "
          f"({100 * wins / n:.1f}% / {100 * losses / n:.1f}% / {100 * pushes / n:.1f}%)")
    print(f"Average bet: ${total_bet / n:.2f}")
    print(f"Illegal action rate: {100 * total_illegal / max(total_actions, 1):.2f}%")
    print(f"Telemetry saved to {args.telemetry_path}")


def main():
    run_baseline(parse_args())


if __name__ == "__main__":
    main()
