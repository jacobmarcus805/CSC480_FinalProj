import torch
import numpy as np
import matplotlib.pyplot as plt
from agent import BlackjackDQN
from environment import SixDeckBlackjack, STATE_DIM


def _masked_action(env, agent, state):
    """Greedy action that respects legal_actions(), mirroring training."""
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = agent(state_tensor).squeeze(0).clone()
    legal = env.legal_actions()
    mask = torch.full_like(q_values, float('-inf'))
    for a in legal:
        mask[a] = 0.0
    q_values = q_values + mask
    return int(q_values.argmax().item())


def plot_training_curve():
    """Chart 1: How the AI learned over time (training-time rewards)."""
    data = np.load("training_telemetry.npy", allow_pickle=True).item()
    rewards = data["rewards"]
    epsilons = data["epsilons"]

    window = 1000
    moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Hands Played')
    ax1.set_ylabel('Average Profit per Hand ($)', color='tab:blue')
    ax1.plot(moving_avg, color='tab:blue', label="Win Rate")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Randomness (Epsilon)', color='tab:red')
    ax2.plot(epsilons[window - 1:], color='tab:red', alpha=0.5, label="Exploration Rate")
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('AI Learning Curve: Profit vs. Exploration')
    plt.tight_layout()
    plt.show()


def plot_neural_weights():
    """Chart 2: Which inputs the network attends to."""
    agent = BlackjackDQN(input_dim=STATE_DIM, output_dim=7)
    agent.load_state_dict(torch.load("blackjack_card_counter_v1.pth"))

    first_layer_weights = agent.fc1.weight.data
    node_importance = torch.abs(first_layer_weights).mean(dim=0).numpy()

    labels = ['Game Phase', 'Player Total', 'Dealer Upcard', 'True Count', 'Is Soft', 'Num Cards']
    colors = ['gray', 'tab:blue', 'tab:blue', 'gold', 'tab:green', 'tab:purple']

    plt.figure(figsize=(10, 5))
    plt.bar(labels, node_importance, color=colors)
    plt.title("Neural Network Attention: Which inputs drive the AI's decisions?")
    plt.ylabel("Average Synaptic Weight (Importance)")
    plt.tight_layout()
    plt.show()


def run_evaluation(num_hands=100000):
    """Plays `num_hands` with the frozen, trained network (no exploration).

    Returns the per-hand profits as a numpy array. This is the source of
    truth for the two evaluation plots below; we only run the simulation once.
    """
    print(f"Starting Final Evaluation Run ({num_hands:,} hands)...")
    env = SixDeckBlackjack()
    agent = BlackjackDQN(input_dim=STATE_DIM, output_dim=7)
    agent.load_state_dict(torch.load("blackjack_card_counter_v1.pth"))
    agent.eval()  # Lock the brain (Epsilon = 0)

    profits = np.zeros(num_hands, dtype=np.float32)
    for i in range(num_hands):
        state = env.reset()
        done = False
        hand_profit = 0.0
        while not done:
            action = _masked_action(env, agent, state)
            state, reward, done, _ = env.step(action)
            hand_profit += reward
        profits[i] = hand_profit

    return profits


def plot_eval_moving_average(profits, window=1000):
    """Chart 3a: Per-hand profit during eval, smoothed.

    Same y-axis units as the training curve, so you can put them side-by-side
    in a slide and the audience can read 'this is the agent's real win rate
    once exploration is turned off'.
    """
    moving_avg = np.convolve(profits, np.ones(window) / window, mode='valid')
    overall = float(profits.mean())

    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg, color='tab:blue')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    plt.axhline(overall, color='tab:red', linestyle=':', linewidth=1.0,
                label=f'Overall mean: ${overall:.2f}/hand')
    plt.title(f'Eval Win Rate (Frozen Agent): {window}-hand Moving Average')
    plt.xlabel('Hands Played')
    plt.ylabel('Average Profit per Hand ($)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def plot_eval_bankroll(profits):
    """Chart 3b: Cumulative bankroll. Useful for 'total damage' framing."""
    bankroll = np.concatenate([[0], np.cumsum(profits)])
    plt.figure(figsize=(10, 5))
    plt.plot(bankroll, color='gold')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    plt.title(f"Cumulative Bankroll Over {len(profits):,} Eval Hands")
    plt.xlabel("Hands Played")
    plt.ylabel("Total Bankroll ($)")
    plt.tight_layout()
    plt.show()


def final_evaluation_run():
    """Backwards-compatible entry point: runs eval and shows both views."""
    profits = run_evaluation(num_hands=100000)
    print(f"Eval mean profit per hand: ${profits.mean():.2f}")
    print(f"Eval total bankroll change: ${profits.sum():.2f}")
    plot_eval_moving_average(profits)
    plot_eval_bankroll(profits)


if __name__ == "__main__":
    # plot_training_curve()
    # plot_neural_weights()
    final_evaluation_run()
