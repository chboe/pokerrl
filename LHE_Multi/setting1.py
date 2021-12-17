from Player import Player
from Hand import LHEHand
from Agents.NFSP_Agent import NFSP_Agent
from Agents.Random_Agent import Random_Agent
import random

USE_TRAINED_MODELS = False

SAVE_INTERVAL = 10_000
MRL_SIZE = 30_000_000
MSL_SIZE = 600_000
RL_LR = 0.01
SL_LR = 0.01
BATCH_SIZE = 256
TARGET_POLICY_UPDATE_INTERVAL = 1000
ANTICIPATORY_PARAM = 0.9
EPS = 0.08

# Setup
agent0 = Random_Agent()
player0 = Player(id=0, agent=agent0)

agent1 = NFSP_Agent(1, SAVE_INTERVAL, MRL_SIZE, MSL_SIZE, RL_LR, SL_LR, BATCH_SIZE, TARGET_POLICY_UPDATE_INTERVAL, ANTICIPATORY_PARAM, EPS)
player1 = Player(id=1, agent=agent1)

agent2 = Random_Agent()
player2 = Player(id=2, agent=agent2)

agent3 = Random_Agent()
player3 = Player(id=3, agent=agent3)

agent4 = Random_Agent()
player4 = Player(id=4, agent=agent4)

agent5 = Random_Agent()
player5 = Player(id=5, agent=agent5)

agent6 = Random_Agent()
player6 = Player(id=6, agent=agent6)

agent7 = Random_Agent()
player7 = Player(id=7, agent=agent7)

agent8 = Random_Agent()
player8 = Player(id=8, agent=agent8)

players_in = [player0, player1, player2, player3, player4, player5, player6, player7, player8]

# Play
episode_counter = 0
while(True):
    num_players = random.randint(2, 9)
    LHE = LHEHand(0.5, random.sample(players_in, num_players))
    LHE.play_hand()

    episode_counter += 1
    if episode_counter % 100 == 0:
        print(f'\nEpisode done: {episode_counter}\n')
        for p in players_in:
            print(f'Player id={p.id}, total winnings={p.total_winnings}, average last 100 = {sum(p.last100)/len(p.last100)}')
