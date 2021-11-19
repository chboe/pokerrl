import numpy as np
from Deck import Deck
from Hand import LHEHand
from RL.Agent import Agent
from Player import Player

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

MRL_SIZE = 30000000
MSL_SIZE = 600000
RL_LR = 0.1
SL_LR = 0.01
BATCH_SIZE = 256
TARGET_POLICY_UPDATE_INTERVAL = 1000
ANTICIPATORY_PARAM = 0.1
EPS = 0.08

player0 = Player(Agent(MRL_SIZE,MSL_SIZE,RL_LR,SL_LR,BATCH_SIZE,TARGET_POLICY_UPDATE_INTERVAL,ANTICIPATORY_PARAM,EPS))
player1 = Player(Agent(MRL_SIZE,MSL_SIZE,RL_LR,SL_LR,BATCH_SIZE,TARGET_POLICY_UPDATE_INTERVAL,ANTICIPATORY_PARAM,EPS))
h = LHEHand(0.5, 0, [player0, player1])
