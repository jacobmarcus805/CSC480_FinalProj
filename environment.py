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


class SixDeckBlackjack:
    def __init__(self):
        # 0: Bet $10, 1: Bet $50, 2: Bet $100
        # 3: Hit, 4: Stand, 5: Double Down, 6: Split (disabled)
        self.action_space = [0, 1, 2, 3, 4, 5, 6]

        # State Array: [Phase, Player_Total, Dealer_Upcard, (10 slots for card counts 2-Ace)]
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

        return self._get_state()

    def legal_actions(self):
        """Returns the list of legal action IDs for the current state.

        Used by both exploration and exploitation so the agent never trains on
        impossible moves. Split is disabled until properly implemented.
        """
        if self.phase == 0:
            return [BET_10, BET_50, BET_100]
        # Phase 1: Hit and Stand are always legal.
        actions = [HIT, STAND]
        # Double Down is only legal on the first two cards.
        if len(self.player_hand) == 2:
            actions.append(DOUBLE)
        # SPLIT (6) is intentionally excluded until implemented.
        return actions

    def step(self, action):
        """The core engine. Takes the AI's action and advances the game."""

        # --- PHASE 0: BETTING ---
        if self.phase == 0:
            if action not in [BET_10, BET_50, BET_100]:
                return self._get_state(), -100, True, {"msg": "Illegal Action: Played during Bet Phase"}

            bet_amounts = {BET_10: 10, BET_50: 50, BET_100: 100}
            self.current_bet = bet_amounts[action]

            self._deal_initial_cards()

            # INSTANT BLACKJACK CHECK
            if self._calculate_total(self.player_hand) == 21:
                # If player gets a natural 21, hand ends immediately.
                dealer_total = self._play_dealer_hand()
                reward = self._determine_winner(21, dealer_total)
                return self._get_state(), reward, True, {"msg": "Natural Blackjack!"}

            self.phase = 1
            return self._get_state(), 0, False, {"msg": "Bet placed"}

        # --- PHASE 1: PLAYING ---
        elif self.phase == 1:
            if action not in self.legal_actions():
                return self._get_state(), -100, True, {"msg": "Illegal Action"}

            if action == HIT:
                self._draw_card(self.player_hand)
                if self._calculate_total(self.player_hand) > 21:
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
        # Deals starting hands. Only the dealer's upcard is tracked in history initially
        self._draw_card(self.player_hand)
        self._draw_card(self.dealer_hand)  # Dealer upcard
        self._draw_card(self.player_hand)
        # We don't draw the dealer's downcard yet to avoid the AI "seeing" it in the deck history

    def _draw_card(self, hand):
        # Pops a card, adds to hand, updates the AIs memory of the deck
        card = self.shoe.pop()
        hand.append(card)

        # Increment the correct slot in the deck history array
        if card == 'A':
            index = 9
        elif card == 'T':
            index = 8
        else:
            index = int(card) - 2

        self.deck_history[index] += 1

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
        # Dealer hits until soft 17
        while self._calculate_total(self.dealer_hand) < 17:
            self._draw_card(self.dealer_hand)
        return self._calculate_total(self.dealer_hand)

    def _determine_winner(self, player_total, dealer_total):
        """Calculates financial reward, including 3:2 blackjack payout."""
        # Natural Blackjack check
        if player_total == 21 and len(self.player_hand) == 2:
            if dealer_total == 21 and len(self.dealer_hand) == 2:
                return 0  # Push
            return self.current_bet * 1.5

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

        cards_dealt = np.sum(self.deck_history)
        cards_remaining = 312 - cards_dealt
        decks_remaining = max(1.0, cards_remaining / 52.0)  # Prevent divide by zero

        return round(running_count / decks_remaining)


# Test
def random_agent_test(env, episodes=1000):
    print("Starting Random Agent Stress Test...")
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = random.choice(env.legal_actions())
            next_state, reward, done, info = env.step(action)
    print(f"Successfully simulated {episodes} hands without crashing!")


env = SixDeckBlackjack()


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
            print(f"Dealer Upcard: {env.dealer_hand[0] if env.dealer_hand else 'Hidden'}")
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
