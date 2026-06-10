#!/usr/bin/env python3
"""Compare DQN training configurations toward break-even play."""

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from agent import BlackjackAgent
from environment import (
    BET_10,
    BET_50,
    BET_100,
    STATE_DIM,
    SixDeckBlackjack,
    legal_action_mask_batch,
)

# Shared hyperparameters (match train.py defaults)
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 1e-4
TAU = 0.005
GRAD_CLIP = 1.0
EPS_START = 1.0
EPS_END = 0.05
DEFAULT_EPS_DECAY = 60_000
ROLLING_WINDOW = 1000
EVAL_EPISODES = 50_000

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

EXPERIMENT_DIR = os.path.join("results", "experiments")
DEFAULT_MODEL_PATH = "blackjack_card_counter_v1.pth"

# names for the four strategies we pit against each other in --compare-eval
DQN_EVAL_NAME = "dqn_betting_play"              # full DQN — bet and play
FIXED_BET_EVAL_NAME = "fixed_betting_play_agent"  # flat $10 bet, DQN plays
HYBRID_BET_EVAL_NAME = "hybrid_count_betting_eval"  # bet by count, DQN plays
RANDOM_BASELINE_NAME = "random_baseline"        # random everything

# per-mode eval telemetry CSVs (written by run_compare_eval_modes)
DQN_EVAL_CSV = os.path.join(EXPERIMENT_DIR, "dqn_eval.csv")
FIXED_BET_EVAL_CSV = os.path.join(EXPERIMENT_DIR, "fixed_bet_eval.csv")
HYBRID_BET_EVAL_CSV = os.path.join(EXPERIMENT_DIR, "hybrid_bet_eval.csv")
RANDOM_BASELINE_EVAL_CSV = os.path.join(EXPERIMENT_DIR, "random_baseline_eval.csv")

# valid eval choices --only to run a single eval mode instead of training
EVAL_MODE_CHOICES = [
    FIXED_BET_EVAL_NAME,
    HYBRID_BET_EVAL_NAME,
    "compare_eval",
]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    train_episodes: int
    eps_decay: int
    description: str


EXPERIMENTS = [
    ExperimentConfig(
        name="baseline_current",
        train_episodes=50_000,
        eps_decay=DEFAULT_EPS_DECAY,
        description="Current hyperparameters, 50k training episodes",
    ),
    ExperimentConfig(
        name="longer_training",
        train_episodes=100_000,
        eps_decay=DEFAULT_EPS_DECAY,
        description="Same hyperparameters, 100k training episodes",
    ),
    ExperimentConfig(
        name="slower_epsilon_decay",
        train_episodes=100_000,
        eps_decay=120_000,
        description="100k episodes with slower epsilon decay (more exploration)",
    ),
]


def parse_args():
    """Parse command-line flags."""
    parser = argparse.ArgumentParser(
        description="Run blackjack DQN training experiments and compare eval results."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train and eval.")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=[cfg.name for cfg in EXPERIMENTS] + EVAL_MODE_CHOICES,
        help="Run a training experiment, eval mode, or compare_eval.",
    )
    parser.add_argument(
        "--compare-eval",
        action="store_true",
        help="Run 50k-hand comparison: DQN, fixed $10, hybrid count, random baseline.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Model weights for DQN play eval modes (no retraining).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Same seed every run so results are comparable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def legal_action_mask(states: torch.Tensor) -> torch.Tensor:
    """Call the env's mask helper and get the result back on the right device."""
    mask_np = legal_action_mask_batch(states.detach().cpu().numpy())
    return torch.from_numpy(mask_np).to(device=states.device)


def epsilon_for_step(steps_done: int, eps_decay: int) -> float:
    """How much random exploration is left after this many steps."""
    return EPS_END + (EPS_START - EPS_END) * np.exp(-steps_done / eps_decay)


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
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    return float(np.mean(rewards[-window:]))


def experiment_paths(name: str) -> dict[str, str]:
    """Where to save the model, training log, and eval log for one experiment."""
    return {
        "model": os.path.join(EXPERIMENT_DIR, f"{name}_model.pth"),
        "train_csv": os.path.join(EXPERIMENT_DIR, f"{name}_training.csv"),
        "eval_csv": os.path.join(EXPERIMENT_DIR, f"{name}_eval.csv"),
    }


def train_experiment(config: ExperimentConfig, seed: int) -> str:
    """Train with one config's settings. Returns path to saved weights."""
    paths = experiment_paths(config.name)
    set_seed(seed)

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

    init_telemetry_csv(paths["train_csv"])
    steps_done = 0
    episode_rewards: list[float] = []

    print(f"\n=== Training: {config.name} ===")
    print(config.description)
    print(
        f"episodes={config.train_episodes:,} | eps_decay={config.eps_decay:,} | "
        f"seed={seed}"
    )

    for episode in range(config.train_episodes):
        state = env.reset()
        true_count_at_bet = env._get_true_count()
        done = False

        total_reward = 0.0
        illegal_action_count = 0
        num_actions = 0
        episode_losses: list[float] = []
        epsilon = EPS_END

        while not done:
            steps_done += 1
            epsilon = epsilon_for_step(steps_done, config.eps_decay)
            action = agent.select_action(state, env.legal_actions(), epsilon)
            next_state, reward, done, info = env.step(action)

            num_actions += 1
            total_reward += reward
            if is_illegal_step(info):
                illegal_action_count += 1

            agent.remember(state, action, reward / 100.0, next_state, done)
            state = next_state

            loss = agent.learn(legal_action_mask)  # copied from train.py
            if loss is not None:
                episode_losses.append(loss)
                agent.soft_update_target(TAU)

        bet_amount = env.current_bet
        outcome = classify_outcome(total_reward)
        avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        episode_rewards.append(total_reward)
        roll_avg = rolling_average(episode_rewards)

        append_telemetry_row(
            paths["train_csv"],
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

        if episode % 5000 == 0:
            print(
                f"  Hand {episode:,}/{config.train_episodes:,} | "
                f"eps={epsilon:.3f} | rolling=${roll_avg:.2f}"
            )

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    agent.save(paths["model"])
    final_roll = rolling_average(episode_rewards)
    print(f"Saved model: {paths['model']}")
    print(f"Saved training telemetry: {paths['train_csv']}")
    print(f"Final training rolling avg: ${final_roll:.2f}/hand")
    return paths["model"]


@dataclass
class EvalSummary:
    name: str
    avg_reward: float
    wins: int
    losses: int
    pushes: int
    avg_bet: float
    illegal_rate: float


def evaluate_experiment(config: ExperimentConfig, seed: int) -> EvalSummary:
    """Play out 50k hands with no exploration and collect stats."""
    paths = experiment_paths(config.name)
    set_seed(seed)

    env = SixDeckBlackjack()
    agent = BlackjackAgent(state_dim=STATE_DIM, action_dim=7, batch_size=BATCH_SIZE, buffer_capacity=1)
    agent.load(paths["model"])
    agent.policy_net.eval()

    init_telemetry_csv(paths["eval_csv"])

    episode_rewards: list[float] = []
    wins = losses = pushes = 0
    total_bet = 0
    total_actions = 0
    total_illegal = 0

    print(f"\n=== Evaluating: {config.name} ({EVAL_EPISODES:,} hands, greedy) ===")

    for episode in range(EVAL_EPISODES):
        state = env.reset()
        true_count_at_bet = env._get_true_count()
        done = False
        total_reward = 0.0
        illegal_action_count = 0
        num_actions = 0

        while not done:
            action = agent.select_action(state, env.legal_actions(), epsilon=0.0)
            state, reward, done, info = env.step(action)
            total_reward += reward
            num_actions += 1
            if is_illegal_step(info):
                illegal_action_count += 1

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
        total_illegal += illegal_action_count

        append_telemetry_row(
            paths["eval_csv"],
            {
                "episode": episode,
                "total_reward": round(total_reward, 2),
                "rolling_avg_reward": round(roll_avg, 2),
                "epsilon": 0.0,
                "outcome": outcome,
                "illegal_action_count": illegal_action_count,
                "num_actions": num_actions,
                "bet_amount": bet_amount,
                "true_count_at_bet": true_count_at_bet,
                "avg_loss": 0.0,
            },
        )

        if episode > 0 and episode % 10_000 == 0:
            print(f"  Eval hand {episode:,}/{EVAL_EPISODES:,} | rolling=${roll_avg:.2f}")

    n = EVAL_EPISODES
    summary = EvalSummary(
        name=config.name,
        avg_reward=float(np.mean(episode_rewards)),
        wins=wins,
        losses=losses,
        pushes=pushes,
        avg_bet=total_bet / n,
        illegal_rate=100.0 * total_illegal / max(total_actions, 1),
    )

    print(f"Saved eval telemetry: {paths['eval_csv']}")
    print_summary(summary, n)
    return summary


def print_summary(summary: EvalSummary, n: int) -> None:
    """Dump win rate, avg bet, etc. for one eval run."""
    print(f"  Average reward/hand: ${summary.avg_reward:.2f}")
    print(
        f"  Win/loss/push: {summary.wins}/{summary.losses}/{summary.pushes} "
        f"({100 * summary.wins / n:.1f}% / {100 * summary.losses / n:.1f}% / "
        f"{100 * summary.pushes / n:.1f}%)"
    )
    print(f"  Average bet: ${summary.avg_bet:.2f}")
    print(f"  Illegal action rate: {summary.illegal_rate:.4f}%")


def hybrid_count_bet_action(true_count: int) -> int:
    """Hard-coded bet sizing for the hybrid eval — not learned."""
    # rough card-counter spread, only here to see if it beats learned betting
    if true_count >= 2:
        return BET_100
    if true_count <= -1:
        return BET_10
    return BET_50


def print_comparison_table(summaries: list[EvalSummary], title: str = "EXPERIMENT COMPARISON") -> None:
    """Print eval results in a table so you can compare at a glance."""
    print("\n" + "=" * 88)
    print(f"{title} ({EVAL_EPISODES:,}-hand eval)")
    print("=" * 88)
    print(
        f"{'Mode':<28} {'Avg $/hand':>11} {'Win%':>7} {'Loss%':>7} "
        f"{'Push%':>7} {'Avg Bet':>8} {'Illegal%':>9}"
    )
    print("-" * 88)
    for s in summaries:
        n = s.wins + s.losses + s.pushes
        print(
            f"{s.name:<28} "
            f"{s.avg_reward:>11.2f} "
            f"{100 * s.wins / n:>6.1f}% "
            f"{100 * s.losses / n:>6.1f}% "
            f"{100 * s.pushes / n:>6.1f}% "
            f"{s.avg_bet:>8.2f} "
            f"{s.illegal_rate:>8.4f}%"
        )
    print("=" * 88)


def run_experiment(config: ExperimentConfig, seed: int) -> EvalSummary:
    """Train, then immediately eval the model we just saved."""
    train_experiment(config, seed)
    return evaluate_experiment(config, seed)


ActionSelector = Callable[
    [SixDeckBlackjack, np.ndarray, Optional[BlackjackAgent], int],
    int,
]


def _dqn_full_action(env, state, agent, _true_count_at_bet):
    """DQN controls everything."""
    return agent.select_action(state, env.legal_actions(), epsilon=0.0)


def _fixed_bet_dqn_play(env, state, agent, _true_count_at_bet):
    """$10 every hand — only tests how well the DQN plays."""
    if env.phase == 0:  # ignore what the network wants to bet
        return BET_10
    return agent.select_action(state, env.legal_actions(), epsilon=0.0)


def _hybrid_bet_dqn_play(env, state, agent, true_count_at_bet):
    """Bet size from true count, hit/stand from the DQN."""
    if env.phase == 0:
        return hybrid_count_bet_action(true_count_at_bet)
    return agent.select_action(state, env.legal_actions(), epsilon=0.0)


def _random_action(env, state, _agent, _true_count_at_bet):
    """Pick a random legal move — our floor to beat."""
    return random.choice(env.legal_actions())


def run_policy_eval(
    name: str,
    select_action: ActionSelector,
    seed: int,
    telemetry_path: str,
    model_path: Optional[str] = None,
    episodes: int = EVAL_EPISODES,
    description: str = "",
) -> EvalSummary:
    """Run N hands with whatever action picker you pass in."""
    agent = None
    # random mode doesn't need a model at all
    if model_path is not None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        agent = BlackjackAgent(
            state_dim=STATE_DIM, action_dim=7, batch_size=BATCH_SIZE, buffer_capacity=1
        )
        agent.load(model_path)
        agent.policy_net.eval()

    set_seed(seed)
    env = SixDeckBlackjack()
    init_telemetry_csv(telemetry_path)

    episode_rewards: list[float] = []
    wins = losses = pushes = 0
    total_bet = 0
    total_actions = 0
    total_illegal = 0

    print(f"\n=== {name} ===")
    if description:
        print(description)
    print(f"episodes={episodes:,} | seed={seed}")

    for episode in range(episodes):
        state = env.reset()
        true_count_at_bet = env._get_true_count()
        done = False
        total_reward = 0.0
        illegal_action_count = 0
        num_actions = 0

        while not done:
            # the callback picks the move — depends which strategy we're running
            action = select_action(env, state, agent, true_count_at_bet)
            state, reward, done, info = env.step(action)
            total_reward += reward
            num_actions += 1
            if is_illegal_step(info):
                illegal_action_count += 1

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
        total_illegal += illegal_action_count

        append_telemetry_row(
            telemetry_path,
            {
                "episode": episode,
                "total_reward": round(total_reward, 2),
                "rolling_avg_reward": round(roll_avg, 2),
                "epsilon": 0.0,
                "outcome": outcome,
                "illegal_action_count": illegal_action_count,
                "num_actions": num_actions,
                "bet_amount": bet_amount,
                "true_count_at_bet": true_count_at_bet,
                "avg_loss": 0.0,
            },
        )

        if episode > 0 and episode % 10_000 == 0:
            print(f"  Hand {episode:,}/{episodes:,} | rolling=${roll_avg:.2f}")

    summary = EvalSummary(
        name=name,
        avg_reward=float(np.mean(episode_rewards)),
        wins=wins,
        losses=losses,
        pushes=pushes,
        avg_bet=total_bet / episodes,
        illegal_rate=100.0 * total_illegal / max(total_actions, 1),
    )

    print(f"Saved telemetry: {telemetry_path}")
    print_summary(summary, episodes)
    return summary


def run_fixed_betting_play_eval(
    model_path: str,
    seed: int,
    telemetry_path: str = FIXED_BET_EVAL_CSV,
    episodes: int = EVAL_EPISODES,
) -> EvalSummary:
    """Fixed $10 bet, DQN plays the hand."""
    return run_policy_eval(
        name=FIXED_BET_EVAL_NAME,
        select_action=_fixed_bet_dqn_play,
        seed=seed,
        telemetry_path=telemetry_path,
        model_path=model_path,
        episodes=episodes,
        description=f"Forced $10 bet | greedy DQN play | model={model_path}",
    )


def run_hybrid_count_betting_eval(
    model_path: str,
    seed: int,
    telemetry_path: str = HYBRID_BET_EVAL_CSV,
    episodes: int = EVAL_EPISODES,
) -> EvalSummary:
    """Count-based betting, DQN plays the hand."""
    return run_policy_eval(
        name=HYBRID_BET_EVAL_NAME,
        select_action=_hybrid_bet_dqn_play,
        seed=seed,
        telemetry_path=telemetry_path,
        model_path=model_path,
        episodes=episodes,
        description=(
            f"Hybrid count bet (TC>=2:$100, TC<=-1:$10, else:$50) | "
            f"greedy DQN play | model={model_path}"
        ),
    )


def run_compare_eval_modes(model_path: str, seed: int) -> list[EvalSummary]:
    """Run all four strategies on the same model and print who's ahead."""
    print("Eval Mode Comparison (no retraining)")
    print(f"Model: {model_path} | seed={seed} | hands={EVAL_EPISODES:,}")

    summaries = [
        run_policy_eval(
            name=DQN_EVAL_NAME,
            select_action=_dqn_full_action,
            seed=seed,
            telemetry_path=DQN_EVAL_CSV,
            model_path=model_path,
            description="DQN betting + DQN play (greedy)",
        ),
        run_fixed_betting_play_eval(model_path, seed),
        run_hybrid_count_betting_eval(model_path, seed),
        run_policy_eval(
            name=RANDOM_BASELINE_NAME,
            select_action=_random_action,
            seed=seed,
            telemetry_path=RANDOM_BASELINE_EVAL_CSV,
            model_path=None,
            description="Random legal actions (bet + play)",
        ),
    ]

    print_comparison_table(
        summaries,
        title="BETTING + PLAY MODE COMPARISON",
    )
    return summaries


def main():
    """Figure out what the user asked for and run it."""
    args = parse_args()
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    if args.compare_eval or args.only == "compare_eval":
        run_compare_eval_modes(args.model_path, args.seed)  # uses existing weights, doesn't train
        return

    if args.only == FIXED_BET_EVAL_NAME:
        run_fixed_betting_play_eval(args.model_path, args.seed)
        return

    if args.only == HYBRID_BET_EVAL_NAME:
        run_hybrid_count_betting_eval(args.model_path, args.seed)
        return

    configs = EXPERIMENTS
    if args.only:
        configs = [cfg for cfg in EXPERIMENTS if cfg.name == args.only]

    print("Blackjack DQN Experiments")
    print(f"Output directory: {EXPERIMENT_DIR}/")
    print(f"Seed: {args.seed}")

    summaries: list[EvalSummary] = []
    for config in configs:
        summaries.append(run_experiment(config, args.seed))

    if len(summaries) > 1:
        print_comparison_table(summaries)


if __name__ == "__main__":
    main()
