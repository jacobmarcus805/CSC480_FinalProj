import numpy as np
import random

# State vector layout (6 features, normalized to roughly [-1, 1] or [0, 1]):
#   0: phase            (0 = betting, 1 = playing)
#   1: player_total/21
#   2: upcard/11
#   3: true_count/10    (clipped to [-1, 1])
#   4: is_soft          (0 or 1, has a usable Ace counted as 11)
#   5: num_player_cards/10
STATE_DIM = 6

# Action IDs
BET_10, BET_50, BET_100 = 0, 1, 2
HIT, STAND, DOUBLE, SPLIT = 3, 4, 5, 6
NUM_ACTIONS = 7

BETTING_ACTIONS = (BET_10, BET_50, BET_100)
PLAYING_ACTIONS = (HIT, STAND)


class SixDeckBlackjack:
    def __init__(self):
        # 0: Bet $10, 1: Bet $50, 2: Bet $100
        # 3: Hit, 4: Stand, 5: Double Down, 6: Split (disabled)
        self.action_space = [0, 1, 2, 3, 4, 5, 6]

        # Hi-Lo card-count history (10 buckets: cards 2-9, 10, Ace)
        self.deck_history = np.zeros(10)
        self.shoe = self._build_shoe()
        self.penetration_limit = 312 * 0.25  # Reshuffle when 75% empty

    def reset(self):
        # Called at the start of every new hand
        if len(self.shoe) < self.penetration_limit:
            self.shoe = self._build_shoe()
            self.deck_history = np.zeros(10)

        self.player_hand = []
        self.dealer_hand = []
        self.current_bet = 0
        self.phase = 0  # Start in Betting Phase
        self.dealer_hole_revealed = False

        return self._get_state()

    def legal_actions(self):
        """Return legal action IDs for the current game state.

        Betting phase: bet sizes only.
        Playing phase: hit and stand always; double only with exactly two cards.
        Split (action 6) is disabled for v1.
        """
        if self.phase == 0:
            return list(BETTING_ACTIONS)

        actions = list(PLAYING_ACTIONS)
        if len(self.player_hand) == 2:
            actions.append(DOUBLE)
        return actions

    def step(self, action):
        """The core engine. Takes the AI's action and advances the game."""

        # --- PHASE 0: BETTING ---
        if self.phase == 0:
            if action not in self.legal_actions():
                return self._get_state(), -100, True, {"msg": "Illegal Action: Played during Bet Phase"}

            bet_amounts = {BET_10: 10, BET_50: 50, BET_100: 100}
            self.current_bet = bet_amounts[action]

            self._deal_initial_cards()

            bj_result = self._check_natural_blackjacks_after_deal()
            if bj_result is not None:
                reward, msg = bj_result
                return self._get_state(), reward, True, {"msg": msg}

            self.phase = 1
            return self._get_state(), 0, False, {"msg": "Bet placed"}

        # --- PHASE 1: PLAYING ---
        elif self.phase == 1:
            if action not in self.legal_actions():
                return self._get_state(), -100, True, {"msg": "Illegal Action"}

            if action == HIT:
                self._draw_card(self.player_hand)
                if self._calculate_total(self.player_hand) > 21:
                    self._reveal_dealer_hole_card()
                    return self._get_state(), -self.current_bet, True, {"msg": "Player Busts"}
                return self._get_state(), 0, False, {"msg": "Player Hits"}

            elif action == STAND:
                dealer_total = self._play_dealer_hand()
                player_total = self._calculate_total(self.player_hand)
                reward = self._determine_winner(player_total, dealer_total)
                return self._get_state(), reward, True, {"msg": "Hand Complete"}

            elif action == DOUBLE:
                # Legality (2-card hand) already enforced by legal_actions check above.
                self.current_bet *= 2
                self._draw_card(self.player_hand)
                if self._calculate_total(self.player_hand) > 21:
                    self._reveal_dealer_hole_card()
                    return self._get_state(), -self.current_bet, True, {"msg": "Double Bust"}

                # If they didn't bust, dealer plays and hand ends.
                dealer_total = self._play_dealer_hand()
                player_total = self._calculate_total(self.player_hand)
                reward = self._determine_winner(player_total, dealer_total)
                return self._get_state(), reward, True, {"msg": "Double Down Complete"}

            # SPLIT is filtered out by legal_actions, so this path should be unreachable.
            return self._get_state(), -100, True, {"msg": "Unhandled action"}

    # Helper

    def _build_shoe(self):
        # Creates a shuffled 6-deck shoe. 'T' represents 10, J, Q, K
        deck = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'T', 'T', 'T', 'A'] * 4
        shoe = deck * 6
        random.shuffle(shoe)
        return shoe

    def _deal_initial_cards(self):
        """Deal two player cards and dealer upcard + hidden hole card."""
        self.dealer_hole_revealed = False
        self._draw_card(self.player_hand)
        self._draw_card(self.dealer_hand)  # Dealer upcard (visible, counted)
        self._draw_card(self.player_hand)
        self._draw_card(self.dealer_hand, track=False)  # Hole card (hidden until reveal)

    def _track_card(self, card):
        """Record a revealed card in the Hi-Lo deck history."""
        if card == 'A':
            index = 9
        elif card == 'T':
            index = 8
        else:
            index = int(card) - 2
        self.deck_history[index] += 1

    def _draw_card(self, hand, track=True):
        """Draw from shoe into hand; optionally add to visible count history."""
        card = self.shoe.pop()
        hand.append(card)
        if track:
            self._track_card(card)

    def _reveal_dealer_hole_card(self):
        """Expose the hole card and add it to deck/count history."""
        if self.dealer_hole_revealed or len(self.dealer_hand) < 2:
            return
        self._track_card(self.dealer_hand[1])
        self.dealer_hole_revealed = True

    def _is_natural_blackjack(self, hand):
        return len(hand) == 2 and self._calculate_total(hand) == 21

    def _check_natural_blackjacks_after_deal(self):
        """Resolve immediate blackjacks after the initial deal.

        Returns ``(reward, msg)`` if the hand ends, else ``None`` to continue.
        The hole card is revealed only when at least one natural is present.
        """
        player_bj = self._is_natural_blackjack(self.player_hand)
        dealer_bj = self._is_natural_blackjack(self.dealer_hand)

        if not player_bj and not dealer_bj:
            return None

        self._reveal_dealer_hole_card()

        if player_bj and dealer_bj:
            return 0, "Push: Both Blackjack"
        if player_bj:
            return self.current_bet * 1.5, "Player Natural Blackjack!"
        return -self.current_bet, "Dealer Natural Blackjack"

    def _calculate_total(self, hand):
        total, _ = self._hand_info(hand)
        return total

    def _hand_info(self, hand):
        """Returns (total, is_soft). is_soft is True iff an Ace is still counted as 11."""
        total = 0
        aces = 0
        for card in hand:
            if card == 'A':
                aces += 1
                total += 11
            elif card == 'T':
                total += 10
            else:
                total += int(card)

        # Downgrade Aces from 11 to 1 if busting
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1

        # Any remaining Ace counted as 11 means the hand is "soft"
        is_soft = aces > 0
        return total, is_soft

    def _play_dealer_hand(self):
        """Reveal hole card, then draw until standing on 17+ (including soft 17)."""
        self._reveal_dealer_hole_card()
        # v1 rule: dealer stands on soft 17 (hits only on 16 or less).
        while self._calculate_total(self.dealer_hand) < 17:
            self._draw_card(self.dealer_hand)
        return self._calculate_total(self.dealer_hand)

    def _determine_winner(self, player_total, dealer_total):
        """Calculate financial reward for a completed non-initial-blackjack hand."""
        if dealer_total > 21:
            return self.current_bet
        if player_total > dealer_total:
            return self.current_bet
        if player_total < dealer_total:
            return -self.current_bet
        return 0  # Tie

    def _get_state(self):
        """Flattens the game data into a normalized 6-node Numpy array.

        Layout: [phase, player_total/21, upcard/11, true_count/10, is_soft, num_cards/10]
        """
        if len(self.player_hand) > 0:
            player_total, is_soft = self._hand_info(self.player_hand)
        else:
            player_total, is_soft = 0, False

        upcard_val = 0
        if len(self.dealer_hand) > 0:
            upcard_str = self.dealer_hand[0]
            if upcard_str == 'A':
                upcard_val = 11
            elif upcard_str == 'T':
                upcard_val = 10
            else:
                upcard_val = int(upcard_str)

        true_count = self._get_true_count()
        # Clip to a reasonable range before normalizing.
        true_count = max(-10, min(10, true_count))

        num_cards = len(self.player_hand)

        return np.array([
            float(self.phase),
            player_total / 21.0,
            upcard_val / 11.0,
            true_count / 10.0,
            1.0 if is_soft else 0.0,
            num_cards / 10.0,
        ], dtype=np.float32)

    def _get_true_count(self):
        """Calculates the Hi-Lo True Count."""
        # Indices 0-4 are cards 2 through 6 (+1 value)
        low_cards = np.sum(self.deck_history[0:5])
        # Indices 8-9 are 10s and Aces (-1 value)
        high_cards = np.sum(self.deck_history[8:10])

        running_count = low_cards - high_cards

        # Use actual shoe size so hidden hole cards affect decks remaining but not count.
        decks_remaining = max(1.0, len(self.shoe) / 52.0)

        return round(running_count / decks_remaining)


def legal_action_mask_batch(states: np.ndarray) -> np.ndarray:
    """Vectorized legal-action mask mirroring ``legal_actions()`` for state rows.

    State layout: [phase, player_total/21, upcard/11, true_count/10,
    is_soft, num_player_cards/10].  Returns a bool array of shape
    (batch, NUM_ACTIONS).
    """
    batch_size = states.shape[0]
    mask = np.zeros((batch_size, NUM_ACTIONS), dtype=bool)

    is_betting = states[:, 0] < 0.5
    is_playing = ~is_betting
    has_two_cards = np.round(states[:, 5] * 10) == 2

    mask[is_betting, BET_10] = True
    mask[is_betting, BET_50] = True
    mask[is_betting, BET_100] = True

    mask[is_playing, HIT] = True
    mask[is_playing, STAND] = True
    mask[is_playing & has_two_cards, DOUBLE] = True

    return mask


def random_agent_test(env, episodes=1000):
    print("Starting Random Agent Stress Test...")
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = random.choice(env.legal_actions())
            next_state, reward, done, info = env.step(action)
    print(f"Successfully simulated {episodes} hands without crashing!")


def human_play_test(env):
    print("Welcome to Terminal Blackjack Debugger")

    while True:
        state = env.reset()
        print("\n--- NEW HAND ---")

        done = False
        while not done:
            # Print the current state so you can verify it
            print(f"Phase: {'Betting' if env.phase == 0 else 'Playing'}")
            print(f"Player Total: {env._calculate_total(env.player_hand)}")
            if env.dealer_hand:
                dealer_show = env.dealer_hand[0]
                if env.dealer_hole_revealed and len(env.dealer_hand) > 1:
                    dealer_show = f"{env.dealer_hand[0]} / {env.dealer_hand[1]}"
                print(f"Dealer: {dealer_show}")
            else:
                print("Dealer: Hidden")
            print(f"Deck History Array: {env.deck_history}")
            print(f"Legal Actions: {env.legal_actions()}")

            # Ask the user for an action
            if env.phase == 0:
                print("Actions: [0]: Bet $10 | [1]: Bet $50 | [2]: Bet $100")
            else:
                print("Actions: [3]: Hit | [4]: Stand | [5]: Double (only on 2 cards)")

            try:
                action = int(input("Enter Action ID: "))
                next_state, reward, done, info = env.step(action)

                print(f"Result: {info['msg']}")
                if done:
                    print(f"Hand Over. Reward: ${reward}")

            except ValueError:
                print("Please enter a valid number.")

        # Pause before the next hand
        cont = input("\nPlay another hand? (y/n): ")
        if cont.lower() != 'y':
            break


def _verify_hole_card_and_blackjack():
    """Sanity-check hidden hole card, reveal timing, and natural blackjack payouts."""
    env = SixDeckBlackjack()

    # Hidden hole card: dealt but not in history until reveal.
    env.reset()
    # Cards are popped from the shoe tail (player, upcard, player, hole).
    env.shoe = ['2', '3', '4', '5', '6', '7', '8', '9', 'A', 'T']
    env.deck_history = np.zeros(10)
    env.current_bet = 10
    env.phase = 0
    env.player_hand = []
    env.dealer_hand = []
    env._deal_initial_cards()
    assert len(env.dealer_hand) == 2
    assert env.dealer_hole_revealed is False
    assert env.deck_history.sum() == 3  # two player cards + dealer upcard only

    # Player natural, dealer not.
    env = SixDeckBlackjack()
    env.reset()
    env.shoe = ['5', '4', '6', 'A', '9', 'T']
    env.deck_history = np.zeros(10)
    _, reward, done, info = env.step(BET_10)
    assert done and reward == 15, (reward, info)
    assert env.dealer_hole_revealed is True
    assert env.deck_history.sum() == 4
    assert env._calculate_total(env.player_hand) == 21
    assert "Player Natural" in info["msg"]

    # Dealer-only natural.
    env = SixDeckBlackjack()
    env.reset()
    env.shoe = ['4', '3', 'A', '5', 'T', '9']
    env.deck_history = np.zeros(10)
    _, reward, done, info = env.step(BET_10)
    assert done and reward == -10, (reward, info)
    assert "Dealer Natural" in info["msg"]

    # Both natural -> push.
    env = SixDeckBlackjack()
    env.reset()
    env.shoe = ['4', '3', 'T', 'A', 'A', 'T']
    env.deck_history = np.zeros(10)
    _, reward, done, info = env.step(BET_50)
    assert done and reward == 0, (reward, info)
    assert "Push" in info["msg"]

    print("hole card and natural blackjack verification passed.")


def _verify_legal_actions():
    """Sanity-check legal_actions() and legal_action_mask_batch() stay in sync."""
    env = SixDeckBlackjack()

    env.reset()
    assert env.legal_actions() == [0, 1, 2], env.legal_actions()

    # Force a non-blackjack hand so betting enters the playing phase.
    env.shoe = ["4", "3", "10", "7", "6", "8"]
    env.deck_history = np.zeros(10)
    env.step(BET_10)
    assert env.phase == 1, env.phase
    assert env.legal_actions() == [3, 4, 5], env.legal_actions()

    env.step(HIT)
    assert env.legal_actions() == [3, 4], env.legal_actions()
    assert SPLIT not in env.legal_actions()

    state = env._get_state()
    mask = legal_action_mask_batch(state.reshape(1, -1))[0]
    assert [i for i in range(NUM_ACTIONS) if mask[i]] == env.legal_actions()

    print("legal_actions() verification passed.")


if __name__ == "__main__":
    _verify_legal_actions()
    _verify_hole_card_and_blackjack()
    env = SixDeckBlackjack()
    random_agent_test(env, episodes=1000)
