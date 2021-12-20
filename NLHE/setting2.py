from Player import Player
from Hand import LHEHand
from Agents.NFSP_Agent import NFSP_Agent
from Agents.Raise_Agent import Raise_Agent
import random

USE_TRAINED_MODELS = False

SAVE_INTERVAL = 10000
MRL_SIZE = 30_000_000
MSL_SIZE = 600_000
RL_LR = 0.01
SL_LR = 0.01
BATCH_SIZE = 256
TARGET_POLICY_UPDATE_INTERVAL = 1000
ANTICIPATORY_PARAM = 0.9
EPS = 0.08
MODEL_TO_LOAD = None

agent2 = NFSP_Agent(99, SAVE_INTERVAL, MRL_SIZE, MSL_SIZE, RL_LR, SL_LR, BATCH_SIZE, TARGET_POLICY_UPDATE_INTERVAL, ANTICIPATORY_PARAM, EPS, MODEL_TO_LOAD)
player2 = Player(id=99, agent=agent2)

# Setup
agent0 = Raise_Agent()
player0 = Player(id=0, agent=agent0)

agent1 = Raise_Agent()
player1 = Player(id=1, agent=agent1)

agent3 = Raise_Agent()
player3 = Player(id=3, agent=agent3)

agent4 = Raise_Agent()
player4 = Player(id=4, agent=agent4)

agent5 = Raise_Agent()
player5 = Player(id=5, agent=agent5)

agent6 = Raise_Agent()
player6 = Player(id=6, agent=agent6)

agent7 = Raise_Agent()
player7 = Player(id=7, agent=agent7)

agent8 = Raise_Agent()
player8 = Player(id=8, agent=agent8)

players_in = [player2, player0, player1, player3, player4, player5, player6, player7, player8]

# Play
episode_counter = 0
while(True):
    LHE = LHEHand(0.5, players_in[:])
    LHE.play_hand()

    episode_counter += 1
    if episode_counter % 100 == 0:
        print(f'\nEpisode done: {episode_counter}\n')
        for p in players_in:
            print(f'Player id={p.id}, total winnings={p.total_winnings}, average last 100 = {sum(p.last100)/len(p.last100)}')
