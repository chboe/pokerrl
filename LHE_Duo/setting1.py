"""Setting for training two NFSP agents against each other.

"""
from Player import Player
from Hand import LHEHand
from Agents.NFSP_Agent import NFSP_Agent
from Agents.Raise_Agent import Raise_Agent
from Agents.Random_Agent import Random_Agent
import time


SAVE_INTERVAL = 2500
MRL_SIZE = 500_000
MSL_SIZE = 1_500_000
RL_LR = 0.1
SL_LR = 0.01
BATCH_SIZE = 256
TARGET_POLICY_UPDATE_INTERVAL = 1000
ANTICIPATORY_PARAM = 0.1
EPS = 0.15
EPS_DECAY = 3_000_000

agent0 = NFSP_Agent(2100, SAVE_INTERVAL, MRL_SIZE, MSL_SIZE, RL_LR, SL_LR, BATCH_SIZE, TARGET_POLICY_UPDATE_INTERVAL, 
                    ANTICIPATORY_PARAM, EPS, EPS_DECAY, None, LEARN=True)
player0 = Player(id=2100, agent=agent0)

agent1 = NFSP_Agent(2101, SAVE_INTERVAL, MRL_SIZE, MSL_SIZE, RL_LR, SL_LR, BATCH_SIZE, TARGET_POLICY_UPDATE_INTERVAL, 
                    ANTICIPATORY_PARAM, EPS, EPS_DECAY, None, LEARN=True)
player1 = Player(id=2101, agent=agent1)

players_in = [player0, player1]
previous_steps = [0, 0]

# Play
before = time.time()
episode_counter = 0
while(True):
    LHE = LHEHand(0.5, players_in[:])
    LHE.play_hand()
    
    episode_counter += 1
    if episode_counter % 1000 == 0:
        now = time.time()
        time_passed = now - before
        print(f'\nEpisode done: {episode_counter}')
        print("Time taken:", time_passed)
        for i, p in enumerate(players_in):
            print(f'\nPlayer id={p.id}, total winnings={p.total_winnings}, average last 100 = {sum(p.last100)/len(p.last100)}')
            steps_taken = p.agent.step_count - previous_steps[i]
            print(f'Player id={p.id} took steps {steps_taken}')
            print(f'Average seconds per step is {time_passed / steps_taken}')
            previous_steps[i] = p.agent.step_count
        before = now
    
    
