# Blackjack Reinforcement Learning Agent

A Deep Q-Network (DQN) agent that learns to play six-deck blackjack through trial and error. The agent chooses bet sizes and play actions (hit, stand, double down) in a custom casino environment, improving its policy from game outcomes over thousands of hands.

## Why This Is a Reinforcement Learning Project

This is not a rule-based blackjack bot. The agent:

- **Interacts with an environment** — each hand is an episode; each bet or play decision is a step.
- **Receives rewards** — wins, losses, pushes, and illegal moves shape future behavior.
- **Learns a policy without labeled “correct” moves** — it discovers when to bet more, hit, stand, or double through experience.
- **Uses experience replay** — past hands are stored and replayed in random batches so learning is stable.
- **Balances exploration and exploitation** — ε-greedy action selection starts random and gradually trusts the neural network.

The goal is to demonstrate that an RL agent can improve decision-making over time, not to implement every casino rule.

## Project Structure

| File                | Purpose                                                                                         |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| `environment.py`    | Six-deck blackjack simulator, state encoding, legal actions                                     |
| `agent.py`          | `ReplayBuffer`, `BlackjackDQN`, and `BlackjackAgent`                                            |
| `train.py`          | Training loop with ε-greedy exploration and CSV telemetry                                       |
| `baseline.py`       | Random legal-action baseline for comparison                                                     |
| `evaluate.py`       | Greedy evaluation of a trained model                                                            |
| `visualize.py`      | Generate presentation plots from CSV telemetry, including the learned-vs-optimal strategy chart |
| `demo.py`           | Live terminal demo of the trained agent                                                         |
| `basic_strategy.py` | Optimal basic-strategy reference tables (S17, 4–8 deck) used by the strategy chart              |

## State Representation

The agent observes a **6-dimensional** normalized vector:

| Index | Feature       | Description                                    |
| ----- | ------------- | ---------------------------------------------- |
| 0     | Phase         | `0` = betting, `1` = playing                   |
| 1     | Player total  | Hand value ÷ 21                                |
| 2     | Dealer upcard | Visible card value ÷ 11                        |
| 3     | True count    | Hi-Lo true count ÷ 10 (clipped)                |
| 4     | Is soft       | `1` if the hand has a usable Ace counted as 11 |
| 5     | Num cards     | Player card count ÷ 10                         |

The dealer hole card is dealt but **not** included in the state until it is revealed at hand resolution.

## Action Space

Seven actions share a single Q-network; illegal actions are masked at selection and during learning.

| ID  | Action                                    |
| --- | ----------------------------------------- |
| 0   | Bet $10                                   |
| 1   | Bet $50                                   |
| 2   | Bet $100                                  |
| 3   | Hit                                       |
| 4   | Stand                                     |
| 5   | Double down (only with exactly two cards) |
| 6   | Split _(disabled in v1)_                  |

**Split is intentionally deferred.** The project focuses on RL learning—bet sizing, hit/stand, and double down—not full casino rule coverage. Split would require multi-hand state and substantially more complexity.

## Reward Function

Rewards are **sparse and monetary** (dollar profit/loss per hand):

| Outcome                  | Reward                |
| ------------------------ | --------------------- |
| Win                      | `+current_bet`        |
| Loss                     | `-current_bet`        |
| Push                     | `0`                   |
| Player natural blackjack | `+1.5 × current_bet`  |
| Dealer natural blackjack | `-current_bet`        |
| Illegal action           | `-100` (episode ends) |

Intermediate play steps return `0` until the hand ends. Rewards are normalized by ÷100 when stored in replay memory.

## DQN Architecture

- **Network:** 6 → 128 → 128 → 7 (ReLU hidden layers, linear output)
- **Algorithm:** Double DQN with soft target updates (τ = 0.005)
- **Loss:** Huber (Smooth L1)
- **Optimizer:** Adam, learning rate `1e-4`
- **Stability:** Gradient clipping, experience replay (200k capacity), batch size 128
- **Exploration:** ε-greedy decay from 1.0 → 0.05 over ~60k steps

Illegal next actions are masked during the bootstrap target computation so the network does not learn Q-values for impossible moves.

## Training Process

1. Reset environment → betting phase.
2. Agent selects a legal action (ε-greedy).
3. Environment advances; reward and next state are returned.
4. Transition is stored in replay memory.
5. After the buffer fills, `BlackjackAgent.learn()` runs a Double DQN update.
6. Target network is softly updated each learn step.
7. Episode telemetry is appended to `training_telemetry.csv`.
8. Checkpoints are saved periodically and at the end.

## Evaluation Metrics

Telemetry CSV columns (training, baseline, and eval):

- Episode reward and rolling average reward
- Epsilon (training only)
- Win / loss / push outcome
- Illegal action count
- Actions per hand, bet amount, true count at bet time
- Average training loss

Summary metrics from `evaluate.py` and `baseline.py`:

- Average reward per hand
- Total bankroll
- Win / loss / push rate
- Average bet size
- Illegal action rate

## Final Results

After 50,000 hands (trained agent vs. random baseline):

| Metric                  | Random Baseline | Trained DQN |
| ----------------------- | --------------- | ----------- |
| **Avg reward / hand**   | −$23.87         | −$1.14      |
| **Win rate**            | 30.3%           | 42.7%       |
| **Loss rate**           | 65.5%           | 48.6%       |
| **Illegal action rate** | 0%              | 0%          |

The trained agent loses far less per hand than random play, wins more often, and never selects illegal actions once exploration decays.

## How to Run

```bash
pip install -r requirements.txt

# Train the agent (50k hands)
python train.py --episodes 50000

# Random baseline for comparison
python baseline.py --episodes 50000

# Evaluate the trained model (greedy, ε=0)
python evaluate.py --episodes 50000 --model-path blackjack_card_counter_v1.pth

# Generate all plots in results
python visualize.py

# Live presentation demo (terminal)
python demo.py --hands 10 --delay 0.5
```

### Useful Options

```bash
python train.py --episodes 50000 --seed 42 --log-interval 1000 --checkpoint-interval 5000
python baseline.py --episodes 50000 --telemetry-path baseline_telemetry.csv
python evaluate.py --episodes 50000 --model-path blackjack_card_counter_v1.pth --telemetry-path eval_telemetry.csv
python visualize.py --output-dir results
```

### Manual Environment Testing

```bash
python environment.py          # verification + random stress test
python -c "from environment import SixDeckBlackjack; ..."  # import-safe (no side effects)
```

## Requirements

- Python 3.10+
- PyTorch, NumPy, Pandas, Matplotlib (see `requirements.txt`)
