from Player import Player
from Hand import LHEHand
from Agents.NFSP_Agent import NFSP_Agent
from Agents.Raise_Agent import Raise_Agent
from Agents.Random_Agent import Random_Agent
import random
import time

USE_TRAINED_MODELS = False

SAVE_INTERVAL = 250_000
MRL_SIZE = 600_000
MSL_SIZE = 3_000_000
RL_LR = 0.01
SL_LR = 0.001
BATCH_SIZE = 256
TARGET_POLICY_UPDATE_INTERVAL = 1000
ANTICIPATORY_PARAM = 0.9
EPS = 0.12
EPS_DECAY = 1_000_000

agent0 = NFSP_Agent(1900, SAVE_INTERVAL, MRL_SIZE, MSL_SIZE, RL_LR, SL_LR, BATCH_SIZE, TARGET_POLICY_UPDATE_INTERVAL, 
                    ANTICIPATORY_PARAM, EPS, EPS_DECAY, None, LEARN=True)
player0 = Player(id=1900, agent=agent0)

agent1 = NFSP_Agent(1901, SAVE_INTERVAL, MRL_SIZE, MSL_SIZE, RL_LR, SL_LR, BATCH_SIZE, TARGET_POLICY_UPDATE_INTERVAL, 
                    ANTICIPATORY_PARAM, EPS, EPS_DECAY, None, LEARN=True)
player1 = Player(id=1901, agent=agent1)

players_in = [player0, player1]

# Play
steps = 0
before = time.time()
episode_counter = 0
while(True):
    
    LHE = LHEHand(0.5, players_in[:])
    LHE.play_hand()

    episode_counter += 1
    if episode_counter % 100 == 0:
        print(f'\nEpisode done: {episode_counter}')
        print(agent1.update_count - steps)
        print(time.time() - before)
        for p in players_in:
            print(f'Player id={p.id}, total winnings={p.total_winnings}, average last 100 = {sum(p.last100)/len(p.last100)}')
        before = time.time()
        steps = agent1.update_count
    
    
