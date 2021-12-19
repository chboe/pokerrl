


from Agents.Verbose_Agent import Verbose_Agent
from Agents.Raise_Agent import Raise_Agent
from Player import Player_Debug as Player
from Hand import LHEHand

agent0 = Verbose_Agent(id='A')
player0 = Player(id=3, agent=agent0)

agent1 = Verbose_Agent(id='B')
player1 = Player(id=8, agent=agent1)

players_in = [player0, player1]

LHE = LHEHand(0.5, players_in[:])
LHE.play_hand()