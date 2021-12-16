from Player import Player
from Hand import LHEHand
from RL.NFSP_Agent import NFSP_Agent
import random

USE_TRAINED_MODELS = False

MRL_SIZE = 30000000
MSL_SIZE = 600000
RL_LR = 0.1
SL_LR = 0.01
BATCH_SIZE = 256
TARGET_POLICY_UPDATE_INTERVAL = 1000
ANTICIPATORY_PARAM = 0.9
EPS = 0.08
EPISODES = 1000

agent0 = NFSP_Agent(MRL_SIZE,MSL_SIZE,RL_LR,SL_LR,BATCH_SIZE,TARGET_POLICY_UPDATE_INTERVAL,ANTICIPATORY_PARAM,EPS)
player0 = Player(id=0, agent=agent0)

agent1 = NFSP_Agent(MRL_SIZE,MSL_SIZE,RL_LR,SL_LR,BATCH_SIZE,TARGET_POLICY_UPDATE_INTERVAL,ANTICIPATORY_PARAM,EPS)
player1 = Player(id=1, agent=agent1)

agent2 = NFSP_Agent(MRL_SIZE,MSL_SIZE,RL_LR,SL_LR,BATCH_SIZE,TARGET_POLICY_UPDATE_INTERVAL,ANTICIPATORY_PARAM,EPS)
player2 = Player(id=2, agent=agent2)

agent3 = NFSP_Agent(MRL_SIZE,MSL_SIZE,RL_LR,SL_LR,BATCH_SIZE,TARGET_POLICY_UPDATE_INTERVAL,ANTICIPATORY_PARAM,EPS)
player3 = Player(id=3, agent=agent3)

agent4 = NFSP_Agent(MRL_SIZE,MSL_SIZE,RL_LR,SL_LR,BATCH_SIZE,TARGET_POLICY_UPDATE_INTERVAL,ANTICIPATORY_PARAM,EPS)
player4 = Player(id=4, agent=agent4)

agent5 = NFSP_Agent(MRL_SIZE,MSL_SIZE,RL_LR,SL_LR,BATCH_SIZE,TARGET_POLICY_UPDATE_INTERVAL,ANTICIPATORY_PARAM,EPS)
player5 = Player(id=5, agent=agent5)

agent6 = NFSP_Agent(MRL_SIZE,MSL_SIZE,RL_LR,SL_LR,BATCH_SIZE,TARGET_POLICY_UPDATE_INTERVAL,ANTICIPATORY_PARAM,EPS)
player6 = Player(id=6, agent=agent6)

agent7 = NFSP_Agent(MRL_SIZE,MSL_SIZE,RL_LR,SL_LR,BATCH_SIZE,TARGET_POLICY_UPDATE_INTERVAL,ANTICIPATORY_PARAM,EPS)
player7 = Player(id=7, agent=agent7)

agent8 = NFSP_Agent(MRL_SIZE,MSL_SIZE,RL_LR,SL_LR,BATCH_SIZE,TARGET_POLICY_UPDATE_INTERVAL,ANTICIPATORY_PARAM,EPS)
player8 = Player(id=8, agent=agent8)

players_in = [player0, player1, player2, player3, player4, player5, player6, player7, player8]

episode_counter = 0

while(True):
    LHE = LHEHand(0.5, players_in[:])
    LHE.play_hand()
    episode_counter += 1
    if episode_counter % 100 == 0:
        print(f'Episode done: {episode_counter}')
        print()
        for p in players_in:
            print(f'Player winnings id={p.id}: {p.total_winnings}.')
        print()
