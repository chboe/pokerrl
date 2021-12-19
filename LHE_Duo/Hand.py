from Deck import Deck
from typing import List
import numpy as np
from Agents.NFSP_Agent import Agent
from Player import Player
from Card import Card
from Score_Detector import HoldemPokerScoreDetector
import random

class ActionEncoding():

    def __init__(self, table_index: int, round: int, num_raises: int,
                action: int):
        self.table_index = table_index
        self.round = round
        self.num_raises = num_raises
        self.action = action

    def __str__(self):
        return f'round={self.round}, seat={self.table_index}, action={ActionEncoding._action_to_str(self.action)}'

    def _action_to_str(action: int):
        return ['raise', 'call', 'fold'][action]


class LHEHand:

    def __init__(self, sb_value: int, players_in: List[Player]):
        #Params
        random.shuffle(players_in)
        self.bb_value = sb_value * 2
        self.sb_value = sb_value
        self.sb_index = 0

        self.players_in = players_in
        self.players_out = []

        #Initialize vars
        self.betting_history: List[ActionEncoding] = []
        self.betting_state = np.zeros((2, 4, 5, 2)) # {num_players, num_rounds, num_bets, num_actions}

        self.flop = np.zeros(52)
        self.turn = np.zeros(52)
        self.river = np.zeros(52)
        self.turn_index = self.sb_index #2P
        self.deck = Deck() # New deck

        # Make agents do pre episode tasks
        for table_index, player in enumerate(self.players_in):
            player.prepare_new_round(table_index)

        
    def play_round(self, round: int):
        self.round = round
        self.num_raises = 0
        self.turn_index = self.sb_index
        self.players_round = 2 # 2P

        if round == 0: # First round is special # 2P
            self.num_raises = 1
        
        while not self.round_over():
            player_to_act = self.players_in[self.turn_index]                      # Find player
            relative_player_state = self.get_relative_player_state(player_to_act) # Get state
            action = player_to_act.get_action(relative_player_state)              # Get action
            self.perform_action(action, player_to_act)                            # Perform action
            self.turn_index = 1 - self.turn_index                                 # rotate turns # 2P


    def perform_action(self, action, player):
        # Force call/fold if num_raises == 4
        if self.num_raises == 4 and action == 0:
            action = 1

        if action == 0: # Raise
            raise_amount = 1 + 1 * (self.round >= 2)
            player.pot = self.players_in[1 - self.turn_index].pot + raise_amount # 2P
            self.betting_state[player.table_index][self.round][self.num_raises][0] = 1
            self.num_raises += 1

        if action == 1: # Call
            player.pot = self.players_in[1 - self.turn_index].pot # 2P
            self.betting_state[player.table_index][self.round][self.num_raises][1] = 1

        if action == 2: # Fold
            #self.betting_state[player.table_index][self.round][self.num_raises][2] = 1
            self.players_out.append(self.players_in.pop(self.turn_index))
            #self.turn_index -= 1

        actionEncoding = ActionEncoding(player.table_index, self.round, self.num_raises, action)
        self.betting_history.append(actionEncoding)

    def check_all_calls(self, current_round_actions):
        calls_to_find = len(self.players_in) - 1
        calls_raises = list(filter(lambda x: x.action != 2, current_round_actions))
        return len(list(filter(lambda x: x.action == 1, calls_raises[-calls_to_find:]))) == calls_to_find

    def round_over(self):
        if len(self.players_in) == 1:
            return True
        current_round_actions = list(filter(lambda x: x.round == self.round, self.betting_history))
        return len(current_round_actions) >= self.players_round and self.check_all_calls(current_round_actions)


    def get_relative_player_state(self, player: Player):
        bh = np.vstack((self.betting_state[player.table_index:], self.betting_state[:player.table_index]))
        own_cards = player.hand
        comm_cards = np.array((self.flop, self.turn, self.river))

        bh = np.ravel(bh)
        own_cards = np.ravel(own_cards)
        comm_cards = np.ravel(comm_cards)

        return np.concatenate((bh, own_cards, comm_cards))


    def deal_cards(self):
        for player in self.players_in: # Deal cards
            player_cards = self.deck.pop_cards(2)
            for card in player_cards:
                player.hand[card.rank - 2 + card.suit*13] = 1.0

    def play_pre_flop(self): #2P next 4 lines
        self.players_in[0].bet(self.sb_value) # sb bet
        self.players_in[1].bet(self.bb_value) # bb bet
        self.betting_state[0][0][0][1] = 1 # Note that sb has "called"
        self.betting_state[1][0][0][0] = 1 # Note that bb has "raised"
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

        winners = 1
        for i in range(1, len(player_scores)):
            if player_scores[0][1] == player_scores[i][1]:
                winners += 1
            else:
                break
        
        for i in range(winners):
            player_scores[i][0].get_result(total_pot/winners - player_scores[i][0].pot)


        for p in player_scores[winners:]:
            p[0].get_result(-p[0].pot)

        for p in self.players_out:
            p.get_result(-p.pot)

    def play_hand(self):
        self.deal_cards()
        self.play_pre_flop()
        self.play_flop()
        self.play_turn()
        self.play_river()

        # Game is over, find winner(s)
        community_cards = self.decode_community_cards()
        player_scores = self.get_player_scores(community_cards)

        # print("--------------------")
        # for a in self.betting_history:
        #     print(a)

        self.distribute_winnings(player_scores)

        
