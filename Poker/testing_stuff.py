import numpy as np
from Deck import Deck
from Hand import LHEHand
from Score_Detector import HoldemPokerScoreDetector
from RL.Agent import Agent
from Player import Player
from Card import Card
from torch import tensor


batch_reward = tensor(np.ones((256)))
batch_next_state = tensor(np.zeros((256,288))) # All batch entries are terminals (full list of zeros)

batch_next_state[0,:] = tensor([1]*288) # Make first batch entry non-terminal
batch_next_state[1,:] = tensor([1]*288) # Make second batch entry non-terminal

print(batch_next_state[0,:].sum().item())
print(batch_next_state[1,:].sum().item())
print(batch_next_state)
max_next_q_values = tensor([3.77]*256)
print(max_next_q_values.size())
terminals = list(map(lambda x: int(x.sum().item() != 0), batch_next_state))


print(max_next_q_values * tensor(terminals))


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
