from Player import Player
from Hand import NLHEHand
from Agents.Random_Agent import Random_Agent
import random

agent0 = Random_Agent()
player0 = Player(id=0, agent=agent0)
agent1 = Random_Agent()
player1 = Player(id=1, agent=agent0)

players_in = [player0, player1]

while(True):
    NLHE = NLHEHand(0.5, players_in)
    NLHE.play_hand()
    raise ValueError("Done :D")