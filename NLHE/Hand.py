from Deck import Deck
from typing import List
import numpy as np
from Agents.NFSP_Agent import Agent
from Player import Player
from Card import Card
from Score_Detector import HoldemPokerScoreDetector
import random
from collections import defaultdict


class ActionEncoding():

    def __init__(self, round: int, action: int, action_value: float, 
            start_stack_size: float, current_stack_pct: float):
        self.round = round
        self.action = action
        self.action_value = action_value
        self.start_stack_size = start_stack_size
        self.current_stack_pct = current_stack_pct

    def __str__(self):
        return f'round={self.round}, action={ActionEncoding._action_to_str(self.action)}, action_value={self.action_value:.4f}, start_stack_size={self.start_stack_size:.2f}, current_stack_pct={self.current_stack_pct:.4f}'

    def _action_to_str(action: int):
        return ['raise', 'call', 'fold'][action]


class BettingHistory():

    def __init__(self, num_players: int):
        self.history = [[] for i in range(num_players)]
        self.num_players = num_players

    def append(self, table_index: int, action: ActionEncoding):
        for i in range(self.num_players):
            self.history[i].append(((table_index - i) % self.num_players, action))
        

class NLHEHand:

    def __init__(self, sb_value: int, players_in: List[Player]):
        #Params
        random.shuffle(players_in)
        self.bb_value = sb_value * 2
        self.sb_value = sb_value
        self.sb_index = 0
        self.num_all_in = [0 for i in range(len(players_in))]

        self.players_in = players_in
        self.players_out = []

        self.bh = BettingHistory(len(players_in))

        self.flop = np.zeros(52)
        self.turn = np.zeros(52)
        self.river = np.zeros(52)
        self.deck = Deck() # New deck

        # Make agents do pre episode tasks
        for table_index, player in enumerate(self.players_in):
            player.prepare_new_round(table_index)

        
    def play_round(self, round: int):
        self.players_round = len(self.players_in)
        self.min_bet = 1
        self.round = round
        if round == 0:
            self.turn_index = (self.players_round + 2) % len(self.players_in)
        else:
            self.turn_index = self.sb_index

        while not self.round_over():
            player_to_act = self.players_in[self.turn_index]                      # Find player
            relative_player_state = self.get_relative_player_state(player_to_act) # Get state
            action = player_to_act.get_action(relative_player_state)              # Get action
            self.perform_action(action, player_to_act)                            # Perform action
            self.turn_index = (self.turn_index + 1) % len(self.players_in)        # rotate turns



    def perform_action(self, action_tuple, player: Player):
        action = action_tuple[0]

        current_stack_size = player.current_stack_size()
        current_stack_pct = player.current_stack_pct
        call_amount = self.max_player_pot - player.pot

        # If player tries to raise, but is short stacked
        short_stacked_raise = False
        if current_stack_size == 0 or (sum(self.num_all_in) >= len(self.players_in) - 1 and action != 2):
            action = 1
        elif action == 2:
            if current_stack_size <= call_amount:
                action = 1
            elif current_stack_size < call_amount + self.min_bet and current_stack_size > call_amount:
                action = 1
                short_stacked_raise = True
            
        if action == 0: # Raise
            raise_to = max(action_tuple[1] * player.start_stack_size + player.pot, self.max_player_pot + self.min_bet)
            raise_amount = player.bet(raise_to - player.pot)
            action_value = raise_amount / player.start_stack_size
            self.min_bet = raise_amount - call_amount
            self.max_player_pot = player.pot
        elif action == 1: # Call
            if not short_stacked_raise:
                actual_call_amount = player.bet(call_amount)
                action_value = actual_call_amount / player.start_stack_size
            else:
                action_value = current_stack_pct
                player.bet(current_stack_size)
                self.max_player_pot = player.pot
        else: # Fold
            self.players_out.append(self.players_in.pop(self.turn_index))
            self.turn_index -= 1
            action_value = 0

        if player.current_stack_pct == 0:
            self.num_all_in[player.table_index] = 1

        actionEncoding = ActionEncoding(self.round, action, action_value, player.start_stack_size, current_stack_pct)
        self.bh.append(player.table_index, actionEncoding)

    def check_all_calls(self, current_round_actions):
        calls_to_find = len(self.players_in) - 1
        calls_raises = list(filter(lambda x: x[1].action != 2, current_round_actions))
        return len(list(filter(lambda x: x[1].action == 1, calls_raises[-calls_to_find:]))) == calls_to_find

    def round_over(self):
        if len(self.players_in) == 1:
            return True
        current_round_actions = list(filter(lambda x: x[1].round == self.round, self.bh.history[0][-9:]))
        return len(current_round_actions) >= self.players_round + 2 * (self.round == 0) and self.check_all_calls(current_round_actions)


    def get_relative_player_state(self, player: Player):
        own_cards = player.hand
        comm_cards = np.array((self.flop, self.turn, self.river))

        own_cards = np.ravel(own_cards)
        comm_cards = np.ravel(comm_cards)
        return self.bh.history[player.table_index], np.concatenate((own_cards, comm_cards))


    def deal_cards(self):
        for player in self.players_in: # Deal cards
            player_cards = self.deck.pop_cards(2)
            for card in player_cards:
                player.hand[card.rank - 2 + card.suit*13] = 1.0

    def play_pre_flop(self):
        # Make smallblind do call
        self.players_in[0].bet(self.sb_value) # sb bet
        s = self.players_in[0].start_stack_size
        self.bh.append(0, ActionEncoding(0, 1, 0.5/s, s, 1))

        self.max_player_pot = 1
        # Make bigblind do raise
        self.players_in[1].bet(self.bb_value) # bb bet
        s = self.players_in[1].start_stack_size
        self.bh.append(1, ActionEncoding(0, 0, 1.0/s, s, 1))
        
        self.play_round(round=0)

    def play_flop(self):
        flopCards = self.deck.pop_cards(3)
        for card in flopCards:
            self.flop[card.rank - 2 + card.suit * 13] = 1
        self.play_round(round=1)

    def play_turn(self):
        turnCard = self.deck.pop_cards(1)[0]
        self.turn[turnCard.rank - 2 + turnCard.suit * 13] = 1
        self.play_round(round=2)
    
    def play_river(self):
        riverCard = self.deck.pop_cards(1)[0]
        self.river[riverCard.rank - 2 + riverCard.suit * 13] = 1
        self.play_round(round=3)

    def decode_community_cards(self):
        community_cards = []
        community_cards += Card.decode_card_array(self.flop)
        community_cards += Card.decode_card_array(self.turn)
        community_cards += Card.decode_card_array(self.river)
        return community_cards

    def get_player_scores(self, community_cards):
        score_detector = HoldemPokerScoreDetector()
        player_scores = []
        for p in self.players_in:
            cards = Card.decode_card_array(p.hand)
            score = score_detector.get_score(cards + community_cards)
            player_scores.append((p, score))

        player_scores = sorted(player_scores, key=(lambda s: s[1].strength))
        player_scores.reverse()
        return player_scores

    def distribute_winnings(self, player_scores):
        total_pot = 0
        for p in self.players_out + self.players_in:
            total_pot += p.pot

        all_players = self.players_in + self.players_out
        num_all_players = len(all_players)

        frozen_pots = [p.pot for p in all_players]
        results = [0 for i in range(num_all_players)]

        # Group player scores by strength
        groups = defaultdict(list)
        for obj in player_scores:
            groups[obj[1].strength].append(obj)

        # Sort groups by strength
        groups = list(groups.values())
        sorted(groups, key=(lambda e: e[0][1].strength)) 

        for i in range(len(groups)):
            take_from_pots = [0 for _ in range(num_all_players)]
            for (p, s) in groups[i]:
                for takee in all_players:
                    ti = takee.table_index
                    take_amount = min(p.pot, frozen_pots[ti] / len(groups[i]))
                    take_from_pots[ti] += take_amount
                    results[p.table_index] += take_amount
                    total_pot -= take_amount

            for i in range(num_all_players):
                frozen_pots[i] -= take_from_pots[i]

            # Break when total_pot has been emptied
            if total_pot < 1e-9:
                break

        for p in all_players:
            p.get_result(results[p.table_index] - p.pot)


    def play_hand(self):
        self.deal_cards()
        self.play_pre_flop()
        self.play_flop()
        self.play_turn()
        self.play_river()

        # Game is over, find winner(s)
        community_cards = self.decode_community_cards()
        player_scores = self.get_player_scores(community_cards)
        self.distribute_winnings(player_scores)
        
