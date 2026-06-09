#!/usr/bin/env python3
"""Terminal demo of the trained blackjack DQN agent."""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

from agent import BlackjackAgent
from environment import (
    BET_10,
    BET_50,
    BET_100,
    HIT,
    STAND,
    DOUBLE,
    STATE_DIM,
    SixDeckBlackjack,
)

ACTION_NAMES = {
    BET_10: "Bet $10",
    BET_50: "Bet $50",
    BET_100: "Bet $100",
    HIT: "Hit",
    STAND: "Stand",
    DOUBLE: "Double Down",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Demo the trained blackjack DQN in the terminal.")
    parser.add_argument("--hands", type=int, default=20, help="Number of hands to play.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="blackjack_card_counter_v1.pth",
        help="Path to trained model weights.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to pause between hands (0 for no delay).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_card(card: str) -> str:
    return "10" if card == "T" else card


def format_hand(cards: list[str]) -> str:
    return " ".join(format_card(c) for c in cards)


def format_dealer(env: SixDeckBlackjack) -> str:
    if not env.dealer_hand:
        return "—"
    upcard = format_card(env.dealer_hand[0])
    if env.dealer_hole_revealed and len(env.dealer_hand) > 1:
        hole = format_card(env.dealer_hand[1])
        total = env._calculate_total(env.dealer_hand)
        return f"{upcard} {hole} (total {total})"
    return f"{upcard} [hole hidden]"


def format_actions(actions: list[int]) -> str:
    return " → ".join(ACTION_NAMES.get(a, f"Action {a}") for a in actions)


def play_demo(args) -> None:
    if not os.path.isfile(args.model_path):
        print(f"Model not found: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    set_seed(args.seed)
    env = SixDeckBlackjack()
    agent = BlackjackAgent(state_dim=STATE_DIM, action_dim=7, batch_size=128, buffer_capacity=1)
    agent.load(args.model_path)
    agent.policy_net.eval()

    bankroll = 0.0
    print(f"Blackjack DQN Demo | {args.hands} hands | model={args.model_path} | seed={args.seed}")
    print("=" * 60)

    for hand_num in range(1, args.hands + 1):
        state = env.reset()
        true_count = env._get_true_count()
        done = False
        hand_reward = 0.0
        actions_taken: list[int] = []

        while not done:
            action = agent.select_action(state, env.legal_actions(), epsilon=0.0)
            actions_taken.append(action)
            state, reward, done, info = env.step(action)
            hand_reward += reward

        bankroll += hand_reward
        player_total = env._calculate_total(env.player_hand)

        print(f"Hand {hand_num}")
        print(f"  Player:     {format_hand(env.player_hand)} (total {player_total})")
        print(f"  Dealer:     {format_dealer(env)}")
        print(f"  True count: {true_count:+d}")
        print(f"  Bet:        ${env.current_bet}")
        print(f"  Actions:    {format_actions(actions_taken)}")
        print(f"  Result:     {info.get('msg', 'Hand over')}")
        print(f"  Reward:     ${hand_reward:+.2f}")
        print(f"  Bankroll:   ${bankroll:+.2f}")
        print("-" * 60)

        if args.delay > 0 and hand_num < args.hands:
            time.sleep(args.delay)

    avg = bankroll / args.hands
    print(f"Demo complete: {args.hands} hands | total bankroll ${bankroll:+.2f} | avg ${avg:+.2f}/hand")


def main():
    play_demo(parse_args())


if __name__ == "__main__":
    main()
