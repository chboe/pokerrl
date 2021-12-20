"""This setting is testing a trained NFSP agent vs an earlier version of itself.

"""
from Player import Player
from Hand import LHEHand
from Agents.NFSP_Agent import NFSP_Agent
from Agents.Raise_Agent import Raise_Agent


SAVE_INTERVAL = 100000
MRL_SIZE = 30_000_000
MSL_SIZE = 600_000
RL_LR = 0.01
SL_LR = 0.01
BATCH_SIZE = 256
TARGET_POLICY_UPDATE_INTERVAL = 1000
ANTICIPATORY_PARAM = 0 # 0 is avgPolicyNetwork, 1 is QNetwork
EPS = 0.08
EPS_DECAY = 1

MODEL_TO_LOAD0 = "Agents/NFSP_Model/id=2000_steps=2000000"
MODEL_TO_LOAD1 = "Agents/NFSP_Model/id=2000_steps=4800000"

agent0 = NFSP_Agent(0, SAVE_INTERVAL, MRL_SIZE, MSL_SIZE, RL_LR, SL_LR, BATCH_SIZE, TARGET_POLICY_UPDATE_INTERVAL, ANTICIPATORY_PARAM, EPS, EPS_DECAY, MODEL_TO_LOAD0, LEARN=False)
player0 = Player(id=0, agent=agent0)

agent1 = NFSP_Agent(1, SAVE_INTERVAL, MRL_SIZE, MSL_SIZE, RL_LR, SL_LR, BATCH_SIZE, TARGET_POLICY_UPDATE_INTERVAL, ANTICIPATORY_PARAM, EPS, EPS_DECAY, MODEL_TO_LOAD1, LEARN=False)
player1 = Player(id=1, agent=agent1)

# Setup
players_in = [player0, player1]

# Play
episode_counter = 0
while(episode_counter < 10000):
    LHE = LHEHand(0.5, players_in[:])
    LHE.play_hand()

    episode_counter += 1
    if episode_counter % 100 == 0:
        print(f'\nEpisode done: {episode_counter}')
        for p in players_in:
            print(f'Player id={p.id}, total winnings={p.total_winnings}, average last 100 = {sum(p.last100)/len(p.last100)}')
