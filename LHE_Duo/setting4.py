"""This setting is testing an already trained NFSP agent vs a Raise Agent.

"""

from Player import Player
from Hand import LHEHand
from Agents.NFSP_Agent import NFSP_Agent
from Agents.Raise_Agent import Raise_Agent

SAVE_INTERVAL = 100000 # Doesnt matter during eval
MRL_SIZE = 100_000 # Doesnt matter during eval
MSL_SIZE = 100_000 # Doesnt matter during eval
RL_LR = 0.01
SL_LR = 0.01
BATCH_SIZE = 256
TARGET_POLICY_UPDATE_INTERVAL = 1000
ANTICIPATORY_PARAM = 1 # 0 is avgPolicyNetwork, 1 is QNetwork
EPS = 0.08
EPS_DECAY = 1_000_000
MODEL_TO_LOAD = "Agents/NFSP_Model/id=2000_steps=7550000"

agent0 = Raise_Agent()
player0 = Player(id=0, agent=agent0)

agent1 = NFSP_Agent(1, SAVE_INTERVAL, MRL_SIZE, MSL_SIZE, RL_LR, SL_LR, BATCH_SIZE, TARGET_POLICY_UPDATE_INTERVAL, ANTICIPATORY_PARAM, EPS, EPS_DECAY, MODEL_TO_LOAD, LEARN=False)
player1 = Player(id=1, agent=agent1)

players_in = [player0, player1]

episode_counter = 0
while(episode_counter < 10_000):
    LHE = LHEHand(0.5, players_in[:])
    LHE.play_hand()

    episode_counter += 1
    if episode_counter % 100 == 0:
        print(f'\nEpisode done: {episode_counter}')
        for p in players_in:
            print(f'Player id={p.id}, total winnings={p.total_winnings}, average last 100 = {sum(p.last100)/len(p.last100)}')