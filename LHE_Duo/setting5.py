"""This setting is testing an already trained NFSP agent vs a Call Agent.

"""

from Player import Player
from Hand import LHEHand
from Agents.NFSP_Agent import NFSP_Agent
from Agents.Call_Agent import Call_Agent

SAVE_INTERVAL = 100000
MRL_SIZE = 30_000_000
MSL_SIZE = 600_000
RL_LR = 0.01
SL_LR = 0.01
BATCH_SIZE = 256
TARGET_POLICY_UPDATE_INTERVAL = 1000
ANTICIPATORY_PARAM = 0.9
EPS = 0.08
EPS_DECAY=1_000_000
MODEL_TO_LOAD = "Agents/NFSP_Model/id=1901_steps=500000"

agent0 = Call_Agent()
player0 = Player(id='Call_Agent', agent=agent0)

agent1 = NFSP_Agent('NFSP_Agent', SAVE_INTERVAL, MRL_SIZE, MSL_SIZE, RL_LR, SL_LR, BATCH_SIZE, TARGET_POLICY_UPDATE_INTERVAL, ANTICIPATORY_PARAM, EPS, EPS_DECAY, MODEL_TO_LOAD, LEARN=False)
player1 = Player(id='NFSP_Agent', agent=agent1)

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