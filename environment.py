import numpy as np
import random

class SixDeckBlackjack:
    def __init__(self):
        # 0: Bet $10, 1: Bet $50, 2: Bet $100
        # 3: Hit, 4: Stand, 5: Double Down, 6: Split
        self.action_space = [0, 1, 2, 3, 4, 5, 6]
        
        # State Array: [Phase, Player_Total, Dealer_Upcard, (10 slots for card counts 2-Ace)]
        self.deck_history = np.zeros(10) 
        self.shoe = self._build_shoe()
        self.penetration_limit = 312 * 0.25 # Reshuffle when 75% empty
        
    def reset(self):
        # Called at the start of every new hand 
        if len(self.shoe) < self.penetration_limit:
            self.shoe = self._build_shoe()
            self.deck_history = np.zeros(10)
            
        self.player_hand = []
        self.dealer_hand = []
        self.current_bet = 0
        self.phase = 0 # Start in Betting Phase
        
        return self._get_state()

    def step(self, action):
        # The core engine. Takes the AIs action and advances the game
        
        # PHASE 0: BETTING
        if self.phase == 0:
            if action not in [0, 1, 2]:
                return self._get_state(), -100, True, {"msg": "Illegal Action: Played during Bet Phase"}
            
            bet_amounts = {0: 10, 1: 50, 2: 100}
            self.current_bet = bet_amounts[action]
            
            self._deal_initial_cards()
            self.phase = 1 # Switch to Playing Phase
            
            return self._get_state(), 0, False, {"msg": "Bet placed"}

        # PHASE 1: PLAYING
        elif self.phase == 1:
            if action not in [3, 4, 5, 6]:
                return self._get_state(), -100, True, {"msg": "Illegal Action: Bet during Play Phase"}
            
            if action == 3: # HIT
                self._draw_card(self.player_hand)
                if self._calculate_total(self.player_hand) > 21:
                    return self._get_state(), -self.current_bet, True, {"msg": "Player Busts"}
                else:
                    return self._get_state(), 0, False, {"msg": "Player Hits"}
                    
            elif action == 4: # STAND
                dealer_total = self._play_dealer_hand()
                player_total = self._calculate_total(self.player_hand)
                reward = self._determine_winner(player_total, dealer_total)
                return self._get_state(), reward, True, {"msg": "Hand Complete"}
            
            # Catch for unimplemented Double/Split for now
            else:
                return self._get_state(), 0, False, {"msg": "Action not fully implemented yet"}

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
        self._draw_card(self.dealer_hand) # Dealer upcard
        self._draw_card(self.player_hand)
        # We don't draw the dealer's downcard yet to avoid the AI "seeing" it in the deck history

    def _draw_card(self, hand):
        # Pops a card, adds to hand, updates the AIs memory of the deck
        card = self.shoe.pop()
        hand.append(card)
        
        # Increment the correct slot in the deck history array
        if card == 'A': index = 9
        elif card == 'T': index = 8
        else: index = int(card) - 2
        
        self.deck_history[index] += 1

    def _calculate_total(self, hand):
        # Sums the hand, handling Aces natively
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
            
        return total

    def _play_dealer_hand(self):
        # Dealer hits until soft 17
        while self._calculate_total(self.dealer_hand) < 17:
            self._draw_card(self.dealer_hand)
        return self._calculate_total(self.dealer_hand)

    def _determine_winner(self, player_total, dealer_total):
        # Calculates financial reward
        if dealer_total > 21: return self.current_bet
        if player_total > dealer_total: return self.current_bet
        if player_total < dealer_total: return -self.current_bet
        return 0 # Push

    def _get_state(self):
        # Flattens the game data into a 1D Numpy Array for the Neural Network
        player_total = self._calculate_total(self.player_hand)
        
        # Convert dealer upcard to an integer for the neural network
        upcard_val = 0
        if len(self.dealer_hand) > 0:
            upcard_str = self.dealer_hand[0]
            if upcard_str == 'A': upcard_val = 11
            elif upcard_str == 'T': upcard_val = 10
            else: upcard_val = int(upcard_str)
        
        state = [self.phase, player_total, upcard_val]
        state.extend(self.deck_history)
        return np.array(state, dtype=np.float32)

# Test
def random_agent_test(env, episodes=1000):
    print("Starting Random Agent Stress Test...")
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if env.phase == 0:
                action = random.choice([0, 1, 2])
            elif env.phase == 1:
                action = random.choice([3, 4]) # Just hitting and standing for the test
                
            next_state, reward, done, info = env.step(action)
    print(f"Successfully simulated {episodes} hands without crashing!")

# Run it
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
            
            # Ask the user for an action
            if env.phase == 0:
                print("Actions: [0]: Bet $10 | [1]: Bet $50 | [2]: Bet $100")
            else:
                print("Actions: [3]: Hit | [4]: Stand | [5]: Double | [6]: Split")
                
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

human_play_test(env)