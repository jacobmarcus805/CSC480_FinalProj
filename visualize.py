#!/usr/bin/env python3
"""Generate presentation plots from training, baseline, and eval CSV telemetry."""

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from environment import HIT, STAND, DOUBLE, STATE_DIM
from basic_strategy import DEALER_UPCARDS, ACTION_TO_CHAR, optimal_action

DEFAULT_TRAINING_CSV = "training_telemetry.csv"
DEFAULT_BASELINE_CSV = "baseline_telemetry.csv"
DEFAULT_EVAL_CSV = "eval_telemetry.csv"
DEFAULT_MODEL_PATH = "blackjack_card_counter_v1.pth"
DEFAULT_OUTPUT_DIR = "results"
ROLLING_WINDOW = 100

HARD_TOTALS = list(range(20, 4, -1))   # 20 down to 5
SOFT_TOTALS = list(range(20, 12, -1))  # soft 20 (A,9) down to soft 13 (A,2)
ACTION_CODE = {HIT: 0, STAND: 1, DOUBLE: 2}
ACTION_CMAP = ListedColormap(["#e74c3c", "#2ecc71", "#f1c40f"])  # H red, S green, D yellow
DIFF_CMAP = ListedColormap(["#dfe6e9", "#c0392b"])               # match gray, mismatch red


def parse_args():
    """Parse command-line flags."""
    parser = argparse.ArgumentParser(description="Plot blackjack RL telemetry from CSV files.")
    parser.add_argument("--training-csv", default=DEFAULT_TRAINING_CSV, help="Training telemetry CSV.")
    parser.add_argument("--baseline-csv", default=DEFAULT_BASELINE_CSV, help="Random baseline CSV.")
    parser.add_argument("--eval-csv", default=DEFAULT_EVAL_CSV, help="Trained agent eval CSV.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Trained model for the strategy chart.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Folder for saved PNG plots.")
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=ROLLING_WINDOW,
        help="Window for rolling illegal-action rate during training.",
    )
    parser.add_argument(
        "--legacy-npy",
        default="training_telemetry.npy",
        help="Optional legacy NumPy telemetry (used only if training CSV is missing).",
    )
    return parser.parse_args()


def load_csv(path: str) -> list[dict] | None:
    """Load a telemetry CSV, or None if the file isn't there."""
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(series: list[dict], key: str) -> np.ndarray:
    """Grab one number column from the CSV rows."""
    return np.array([float(row[key]) for row in series], dtype=np.float64)


def to_int(series: list[dict], key: str) -> np.ndarray:
    """Grab one int column from the CSV rows."""
    return np.array([int(float(row[key])) for row in series], dtype=np.int64)  # everything comes in as a string


def to_str(series: list[dict], key: str) -> list[str]:
    """Grab one text column from the CSV rows."""
    return [row[key] for row in series]


def ensure_output_dir(path: str) -> None:
    """Make the results folder if needed."""
    os.makedirs(path, exist_ok=True)


def save_figure(fig, output_dir: str, filename: str) -> str:
    """Save the plot as a PNG and close it so we don't leak memory."""
    ensure_output_dir(output_dir)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def load_training_data(args) -> list[dict] | None:
    """Prefer the CSV; fall back to old .npy format if that's all we have."""
    rows = load_csv(args.training_csv)
    if rows is not None:
        return rows

    # older runs saved rewards/epsilons as a numpy dict instead of CSV
    if os.path.isfile(args.legacy_npy):
        data = np.load(args.legacy_npy, allow_pickle=True).item()
        rewards = data["rewards"]
        epsilons = data["epsilons"]
        rows = []
        running = []
        for episode, (reward, epsilon) in enumerate(zip(rewards, epsilons)):
            running.append(reward)
            rows.append({
                "episode": str(episode),
                "total_reward": str(reward),
                "rolling_avg_reward": str(np.mean(running[-1000:])),
                "epsilon": str(epsilon),
                "outcome": "win" if reward > 0 else ("loss" if reward < 0 else "push"),
                "illegal_action_count": "0",
                "num_actions": "1",
                "bet_amount": "0",
                "true_count_at_bet": "0",
                "avg_loss": "0",
            })
        print(f"Using legacy telemetry: {args.legacy_npy}")
        return rows

    return None


def plot_training_rolling_reward(training_rows: list[dict], output_dir: str) -> str | None:
    """Is the agent getting better over time?"""
    episodes = to_int(training_rows, "episode")
    rolling = to_float(training_rows, "rolling_avg_reward")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rolling, color="tab:blue", linewidth=1.5, label="Rolling avg reward")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("Training: Rolling Average Reward per Hand")
    ax.set_xlabel("Episode (Hand)")
    ax.set_ylabel("Average Profit per Hand ($)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, output_dir, "01_training_rolling_reward.png")


def plot_training_bankroll(training_rows: list[dict], output_dir: str) -> str | None:
    """Running total $ won or lost across all training hands."""
    episodes = to_int(training_rows, "episode")
    rewards = to_float(training_rows, "total_reward")
    bankroll = np.cumsum(rewards)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, bankroll, color="tab:green", linewidth=1.5, label="Cumulative bankroll")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("Training: Cumulative Bankroll")
    ax.set_xlabel("Episode (Hand)")
    ax.set_ylabel("Total Bankroll ($)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, output_dir, "02_training_cumulative_bankroll.png")


def plot_epsilon_decay(training_rows: list[dict], output_dir: str) -> str | None:
    """When did the agent stop exploring randomly?"""
    episodes = to_int(training_rows, "episode")
    epsilon = to_float(training_rows, "epsilon")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, epsilon, color="tab:red", linewidth=1.5, label="Epsilon")
    ax.set_title("Training: Epsilon Decay (Exploration Rate)")
    ax.set_xlabel("Episode (Hand)")
    ax.set_ylabel("Epsilon")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, output_dir, "03_epsilon_decay.png")


def plot_agent_vs_baseline_reward(
    eval_rows: list[dict] | None,
    baseline_rows: list[dict] | None,
    output_dir: str,
) -> str | None:
    """Who loses less per hand — us or random play?"""
    if eval_rows is None or baseline_rows is None:
        return None

    eval_mean = float(np.mean(to_float(eval_rows, "total_reward")))
    baseline_mean = float(np.mean(to_float(baseline_rows, "total_reward")))

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Trained Agent", "Random Baseline"]
    values = [eval_mean, baseline_mean]
    colors = ["tab:blue", "tab:orange"]
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("Average Reward per Hand: Trained Agent vs Random Baseline")
    ax.set_ylabel("Average Profit per Hand ($)")
    ax.bar_label(bars, fmt="$%.2f", padding=3)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, output_dir, "04_agent_vs_baseline_reward.png")


def outcome_rates(rows: list[dict]) -> dict[str, float]:
    """What % of hands ended in a win, loss, or push."""
    outcomes = to_str(rows, "outcome")
    n = len(outcomes)
    return {
        "win": 100.0 * outcomes.count("win") / n,
        "loss": 100.0 * outcomes.count("loss") / n,
        "push": 100.0 * outcomes.count("push") / n,
    }


def plot_win_loss_push_comparison(
    eval_rows: list[dict] | None,
    baseline_rows: list[dict] | None,
    output_dir: str,
) -> str | None:
    """Win/loss/push breakdown — agent vs random."""
    if eval_rows is None or baseline_rows is None:
        return None

    eval_rates = outcome_rates(eval_rows)
    baseline_rates = outcome_rates(baseline_rows)
    categories = ["Win", "Loss", "Push"]
    keys = ["win", "loss", "push"]
    eval_vals = [eval_rates[k] for k in keys]
    baseline_vals = [baseline_rates[k] for k in keys]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    # nudge the bars left/right so they don't stack on top of each other
    ax.bar(x - width / 2, eval_vals, width, label="Trained Agent", color="tab:blue")
    ax.bar(x + width / 2, baseline_vals, width, label="Random Baseline", color="tab:orange")
    ax.set_title("Win / Loss / Push Rate Comparison")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x, categories)
    ax.set_ylim(0, max(max(eval_vals), max(baseline_vals)) * 1.15 + 1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, output_dir, "05_win_loss_push_comparison.png")


def plot_illegal_action_rate(training_rows: list[dict], output_dir: str, window: int) -> str | None:
    """Illegal moves per step — should drop to ~0 as exploration fades."""
    illegal = to_float(training_rows, "illegal_action_count")
    actions = to_float(training_rows, "num_actions")
    episodes = to_int(training_rows, "episode")

    # divide illegal count by total actions that hand (not all hands have the same length)
    per_step_rate = np.divide(
        illegal,
        actions,
        out=np.zeros_like(illegal),
        where=actions > 0,
    )

    if len(per_step_rate) >= window:
        kernel = np.ones(window) / window
        rolling_rate = np.convolve(per_step_rate, kernel, mode="valid")  # smooth it out
        rolling_episodes = episodes[window - 1:]  # convolve eats the first window-1 points
    else:
        rolling_rate = per_step_rate
        rolling_episodes = episodes

    rolling_pct = 100.0 * rolling_rate

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rolling_episodes, rolling_pct, color="tab:purple", linewidth=1.5)
    ax.set_title(f"Training: Illegal Action Rate ({window}-Episode Rolling Average)")
    ax.set_xlabel("Episode (Hand)")
    ax.set_ylabel("Illegal Action Rate (%)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, output_dir, "06_illegal_action_rate.png")


def average_bet_by_true_count(rows: list[dict]) -> tuple[list[int], list[float]]:
    """Average bet size grouped by true count when the bet was placed."""
    buckets: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        tc = int(float(row["true_count_at_bet"]))  # e.g. all TC=+2 hands go in one bucket
        buckets[tc].append(float(row["bet_amount"]))
    counts = sorted(buckets.keys())
    avg_bets = [float(np.mean(buckets[c])) for c in counts]
    return counts, avg_bets


def plot_average_bet_by_true_count(
    eval_rows: list[dict] | None,
    baseline_rows: list[dict] | None,
    output_dir: str,
) -> str | None:
    """Does the agent bet bigger when the count is good?"""
    if eval_rows is None and baseline_rows is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False

    if eval_rows is not None:
        counts, bets = average_bet_by_true_count(eval_rows)
        ax.plot(counts, bets, marker="o", linewidth=1.5, label="Trained Agent", color="tab:blue")
        plotted = True

    if baseline_rows is not None:
        counts, bets = average_bet_by_true_count(baseline_rows)
        ax.plot(counts, bets, marker="s", linewidth=1.5, label="Random Baseline", color="tab:orange")
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_title("Average Bet Size by True Count at Betting Time")
    ax.set_xlabel("True Count Bucket")
    ax.set_ylabel("Average Bet ($)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, output_dir, "07_average_bet_by_true_count.png")


def _make_state(player_total: int, upcard_val: int, is_soft: bool, true_count: float) -> np.ndarray:

    return np.array([
        1.0,                       
        player_total / 21.0,
        upcard_val / 11.0,
        true_count / 10.0,
        1.0 if is_soft else 0.0,
        2 / 10.0,                  
    ], dtype=np.float32)


def _build_strategy_grids(agent, totals, is_soft, true_count):
    """Sweep the agent over a grid of two-card states; return learned/optimal action grids."""
    rows, cols = len(totals), len(DEALER_UPCARDS)
    learned = np.zeros((rows, cols), dtype=int)
    optimal = np.zeros((rows, cols), dtype=int)
    for r, total in enumerate(totals):
        for c, up in enumerate(DEALER_UPCARDS):
            state = _make_state(total, up, is_soft, true_count)
            learned[r, c] = agent.select_action(state, [HIT, STAND, DOUBLE], epsilon=0.0)
            optimal[r, c] = optimal_action(total, is_soft, up)
    return learned, optimal


def _upcard_labels():
    return [("A" if v == 11 else str(v)) for v in DEALER_UPCARDS]


def _row_labels(totals, is_soft):
    if is_soft:
        return [f"A,{t - 11}  ({t})" for t in totals]  # soft 18 = A,7, etc.
    return [str(t) for t in totals]


def _draw_policy(ax, action_grid, totals, is_soft, title):
    codes = np.vectorize(ACTION_CODE.get)(action_grid)
    ax.imshow(codes, cmap=ACTION_CMAP, vmin=0, vmax=2, aspect="auto")
    for r in range(action_grid.shape[0]):
        for c in range(action_grid.shape[1]):
            ax.text(c, r, ACTION_TO_CHAR[action_grid[r, c]],
                    ha="center", va="center", fontsize=8, fontweight="bold", color="black")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(DEALER_UPCARDS)))
    ax.set_xticklabels(_upcard_labels(), fontsize=8)
    ax.set_yticks(range(len(totals)))
    ax.set_yticklabels(_row_labels(totals, is_soft), fontsize=8)
    ax.set_xlabel("Dealer upcard", fontsize=9)


def _draw_diff(ax, learned, optimal, totals, is_soft, title):
    mism = (learned != optimal).astype(int)
    ax.imshow(mism, cmap=DIFF_CMAP, vmin=0, vmax=1, aspect="auto")
    for r in range(mism.shape[0]):
        for c in range(mism.shape[1]):
            if mism[r, c]:  
                ax.text(c, r, f"{ACTION_TO_CHAR[learned[r, c]]}/{ACTION_TO_CHAR[optimal[r, c]]}",
                        ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(DEALER_UPCARDS)))
    ax.set_xticklabels(_upcard_labels(), fontsize=8)
    ax.set_yticks(range(len(totals)))
    ax.set_yticklabels(_row_labels(totals, is_soft), fontsize=8)
    ax.set_xlabel("Dealer upcard", fontsize=9)


def plot_strategy_chart(model_path: str, output_dir: str, true_count: float = 0.0) -> str | None:
    """Render learned vs. optimal play strategy heatmaps + match score, or None if no model."""
    if not os.path.isfile(model_path):
        return None

    from agent import BlackjackAgent

    agent = BlackjackAgent(state_dim=STATE_DIM, action_dim=7, batch_size=128, buffer_capacity=1)
    agent.load(model_path)
    agent.policy_net.eval()

    h_learn, h_opt = _build_strategy_grids(agent, HARD_TOTALS, False, true_count)
    s_learn, s_opt = _build_strategy_grids(agent, SOFT_TOTALS, True, true_count)

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    _draw_policy(axes[0, 0], h_learn, HARD_TOTALS, False, "Learned — Hard totals")
    _draw_policy(axes[0, 1], h_opt, HARD_TOTALS, False, "Optimal — Hard totals")
    _draw_diff(axes[0, 2], h_learn, h_opt, HARD_TOTALS, False, "Disagreements — Hard")
    _draw_policy(axes[1, 0], s_learn, SOFT_TOTALS, True, "Learned — Soft totals")
    _draw_policy(axes[1, 1], s_opt, SOFT_TOTALS, True, "Optimal — Soft totals")
    _draw_diff(axes[1, 2], s_learn, s_opt, SOFT_TOTALS, True, "Disagreements — Soft")
    axes[0, 0].set_ylabel("Player total", fontsize=9)
    axes[1, 0].set_ylabel("Player soft total", fontsize=9)

    hard_match = float(np.mean(h_learn == h_opt))
    soft_match = float(np.mean(s_learn == s_opt))
    total_match = (np.sum(h_learn == h_opt) + np.sum(s_learn == s_opt)) / (h_learn.size + s_learn.size)

    legend = [
        Patch(facecolor="#e74c3c", label="Hit"),
        Patch(facecolor="#2ecc71", label="Stand"),
        Patch(facecolor="#f1c40f", label="Double"),
        Patch(facecolor="#c0392b", label="Disagreement (learned/optimal)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(
        f"Learned DQN policy vs. optimal basic strategy  (true count = {true_count:g})\n"
        f"Overall match: {total_match:.1%}   (hard {hard_match:.1%}, soft {soft_match:.1%})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out_path = save_figure(fig, output_dir, "08_strategy_chart.png")
    print(f"  strategy chart match vs optimal: {total_match:.1%} "
          f"(hard {hard_match:.1%}, soft {soft_match:.1%})")
    return out_path


def main():
    """Make plots for whatever CSVs exist, skip the rest."""
    args = parse_args()
    created: list[str] = []
    skipped: list[str] = []

    training_rows = load_training_data(args)
    baseline_rows = load_csv(args.baseline_csv)
    eval_rows = load_csv(args.eval_csv)

    if training_rows is not None:
        created.append(plot_training_rolling_reward(training_rows, args.output_dir))
        created.append(plot_training_bankroll(training_rows, args.output_dir))
        created.append(plot_epsilon_decay(training_rows, args.output_dir))
        created.append(plot_illegal_action_rate(training_rows, args.output_dir, args.rolling_window))
    else:
        skipped.extend([
            "01_training_rolling_reward.png (missing training CSV)",
            "02_training_cumulative_bankroll.png (missing training CSV)",
            "03_epsilon_decay.png (missing training CSV)",
            "06_illegal_action_rate.png (missing training CSV)",
        ])

    path = plot_agent_vs_baseline_reward(eval_rows, baseline_rows, args.output_dir)
    if path:
        created.append(path)
    else:
        skipped.append("04_agent_vs_baseline_reward.png (missing eval or baseline CSV)")

    path = plot_win_loss_push_comparison(eval_rows, baseline_rows, args.output_dir)
    if path:
        created.append(path)
    else:
        skipped.append("05_win_loss_push_comparison.png (missing eval or baseline CSV)")

    path = plot_average_bet_by_true_count(eval_rows, baseline_rows, args.output_dir)
    if path:
        created.append(path)
    else:
        skipped.append("07_average_bet_by_true_count.png (missing eval and baseline CSV)")

    path = plot_strategy_chart(args.model_path, args.output_dir)
    if path:
        created.append(path)
    else:
        skipped.append(f"08_strategy_chart.png (missing model: {args.model_path})")

    created = [p for p in created if p]  # drop any None returns from missing data

    print(f"Saved {len(created)} plot(s) to {args.output_dir}/")
    for path in created:
        print(f"  - {path}")

    if skipped:
        print("Skipped:")
        for item in skipped:
            print(f"  - {item}")

    if not created:
        print("No plots created. Run train.py, baseline.py, and evaluate.py first.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
