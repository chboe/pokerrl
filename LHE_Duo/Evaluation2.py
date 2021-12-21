from Player import Player
from Hand import LHEHand
from Agents.NFSP_Agent import NFSP_Agent
from Agents.Raise_Agent import Raise_Agent
from Agents.Call_Agent import Call_Agent
import csv


SAVE_INTERVAL = 100000 # Doesnt matter during eval
MRL_SIZE = 100_000 # Doesnt matter during eval
MSL_SIZE = 100_000 # Doesnt matter during eval
RL_LR = 0.01
SL_LR = 0.01
BATCH_SIZE = 256
TARGET_POLICY_UPDATE_INTERVAL = 1000
ANTICIPATORY_PARAM = 0 # 0 is avgPolicyNetwork, 1 is QNetwork
if ANTICIPATORY_PARAM == 1:
    network = 'Q'
elif ANTICIPATORY_PARAM == 0:
    network = 'A'
else:
    network = f'mixed_{ANTICIPATORY_PARAM}'

EPS = 0.00
EPS_DECAY = 1


model_id = 2100
step_size = 2500
limit = 237500
header = ['steps', 'winnings']

# Play against Call Agent
with open(f'Evaluation/model_id={model_id}_Network={network}_versus=Call_Agent.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for steps in range(step_size, limit+1, step_size):
        MODEL_TO_LOAD = f'Agents/NFSP_Model/id={model_id}_steps={steps}'
        agent0 = Call_Agent()
        player0 = Player(id=0, agent=agent0)
        agent1 = NFSP_Agent(1, SAVE_INTERVAL, MRL_SIZE, MSL_SIZE, RL_LR, SL_LR, BATCH_SIZE, TARGET_POLICY_UPDATE_INTERVAL, ANTICIPATORY_PARAM, EPS, EPS_DECAY, MODEL_TO_LOAD, LEARN=False)
        player1 = Player(id=1, agent=agent1)
        players_in = [player0, player1]

        for episodes in range(10000):
            LHE = LHEHand(0.5, players_in[:])
            LHE.play_hand()

        writer.writerow([steps, player1.total_winnings])

# Play against Raise Agent
with open(f'Evaluation/model_id={model_id}_Network={network}_versus=Raise_Agent.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for steps in range(step_size, limit+1, step_size):
        MODEL_TO_LOAD = f'Agents/NFSP_Model/id={model_id}_steps={steps}'
        agent0 = Raise_Agent()
        player0 = Player(id=0, agent=agent0)
        agent1 = NFSP_Agent(1, SAVE_INTERVAL, MRL_SIZE, MSL_SIZE, RL_LR, SL_LR, BATCH_SIZE, TARGET_POLICY_UPDATE_INTERVAL, ANTICIPATORY_PARAM, EPS, EPS_DECAY, MODEL_TO_LOAD, LEARN=False)
        player1 = Player(id=1, agent=agent1)
        players_in = [player0, player1]

        for episodes in range(10000):
            LHE = LHEHand(0.5, players_in[:])
            LHE.play_hand()

        writer.writerow([steps, player1.total_winnings])

# Play against earlier versions of itself
with open(f'Evaluation/model_id={model_id}_Network={network}_versus=Itself.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    MODEL_TO_LOAD1 = f'Agents/NFSP_Model/id={model_id}_steps={limit}'
    agent1 = NFSP_Agent(1, SAVE_INTERVAL, MRL_SIZE, MSL_SIZE, RL_LR, SL_LR, BATCH_SIZE, TARGET_POLICY_UPDATE_INTERVAL, ANTICIPATORY_PARAM, EPS, EPS_DECAY, MODEL_TO_LOAD1, LEARN=False)
    player1 = Player(id=1, agent=agent1)
    
    for steps in range(step_size, limit, step_size):
        MODEL_TO_LOAD0 = f'Agents/NFSP_Model/id={model_id}_steps={steps}'
        agent0 = NFSP_Agent(0, SAVE_INTERVAL, MRL_SIZE, MSL_SIZE, RL_LR, SL_LR, BATCH_SIZE, TARGET_POLICY_UPDATE_INTERVAL, ANTICIPATORY_PARAM, EPS, EPS_DECAY, MODEL_TO_LOAD0, LEARN=False)
        player0 = Player(id=0, agent=agent0)

        players_in = [player0, player1]
        for episodes in range(10000):
            LHE = LHEHand(0.5, players_in[:])
            LHE.play_hand()

        writer.writerow([steps, player1.total_winnings])
        player1.total_winnings = 0 # Reset player winnings before next opponent


