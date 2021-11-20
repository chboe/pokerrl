from Player import Player
from Hand import LHEHand
from RL.Agent import Agent

USE_TRAINED_MODELS = False
PLAYER_COUNT = 2

MRL_SIZE = 30000000
MSL_SIZE = 600000
RL_LR = 0.1
SL_LR = 0.01
BATCH_SIZE = 256
TARGET_POLICY_UPDATE_INTERVAL = 1000
ANTICIPATORY_PARAM = 0.1
EPS = 0.08
EPISODES = 1000

agent0 = Agent(MRL_SIZE,MSL_SIZE,RL_LR,SL_LR,BATCH_SIZE,TARGET_POLICY_UPDATE_INTERVAL,ANTICIPATORY_PARAM,EPS)
player0 = Player(id=0, agent=agent0)

agent1 = Agent(MRL_SIZE,MSL_SIZE,RL_LR,SL_LR,BATCH_SIZE,TARGET_POLICY_UPDATE_INTERVAL,ANTICIPATORY_PARAM,EPS)
player1 = Player(id=1, agent=agent1)

episode_counter = 0
wins = [0,0,0]

while(True):
    winner = LHEHand(0.5, episode_counter%2, [player0, player1]).winner
    wins[winner] += 1
    episode_counter += 1
    if episode_counter % 200 == 0:
        print(wins)
