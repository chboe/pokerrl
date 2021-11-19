import numpy as np
from Deck import Deck
from Hand import LHEHand
from Score_Detector import HoldemPokerScoreDetector
from RL.Agent import Agent
from Player import Player
from Card import Card

# array = [[0,0,0],
#          [1,1,1],
#          [2,2,2],
#          [3,3,3],
#          [4,4,4],
#          [5,5,5]]
#
# array = np.array(array)
# x = array[5:]
# y = array[:5]
# print(np.vstack((x,y)))


# flop = np.zeros(52)
# turn = np.zeros(52)
# river = np.zeros(52)
#
#
# deck = Deck()
# flopCards = deck.pop_cards(52)
# print(flopCards)
#
#
# for card in flopCards:
#     flop[card.rank-2+card.suit*13] = 1
#
# print(np.ravel((flop, turn, river)))

# hesd = HoldemPokerScoreDetector()
#
#
# deck = Deck()
# player1 = deck.pop_cards(2)
# print("PLAYER 1:")
# for card in player1:
#     print(Card.SUITS[card.suit],Card.RANKS[card.rank])
# print()
# print("PLAYER 2:")
# player2 = deck.pop_cards(2)
# for card in player2:
#     print(Card.SUITS[card.suit],Card.RANKS[card.rank])
# print()
#
# communityCards = deck.pop_cards(5)
# player1 += communityCards
# player2 += communityCards
#
# for card in communityCards:
#     print(Card.SUITS[card.suit],Card.RANKS[card.rank])
#
#
# p1_score = hesd.get_score(player1)
# p2_score = hesd.get_score(player2)
# print(p1_score.cmp(p2_score))
