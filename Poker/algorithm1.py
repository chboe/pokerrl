import math
import random
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.autograd import Variable
from types import ReplayMemory, Network


#HYPER PARAMS
MRL_SIZE = 600000
MSL_SIZE = 3000000
RL_LR = 0.1
SL_LR = 0.01
BATCH_SIZE = 256
TARGET_POLICY_UPDATE_INTERVAL = 1000
ANTICIPATORY_PARAM = 0.1
EPISODES = 1000
EPS_START = 0.08
EPS_END = 0
EPS_DECAY = EPISODES

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

#Initializing everything
steps_done = 0
update_count = 0
mrl = ReplayMemory(MRL_SIZE)
msl = ReplayMemory(MSL_SIZE)
averagePolicyNetwork = Network()
qNetwork = Network()
targetPolicyNetwork = Network()
targetPolicyNetwork.load_state_dict(qNetwork.state_dict())
qNetworkOptimizer = optim.SGD(qNetwork.parameters(), RL_LR)
averagePolicyNetworkOptimizer = optim.SGD(averagePolicyNetwork.parameters(), SL_LR)


def select_action(currentPolicy, state, usingQpolicy):
    global steps_done
    CURR_EPS = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if random.uniform(0, 1) > CURR_EPS or usingQpolicy:
        return currentPolicy(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(3)]])


def learnAveragePolicyNetwork():
    if len(msl) < BATCH_SIZE:
        return

    transitions = msl.sample(BATCH_SIZE)
    batch_state, batch_action = zip(*transitions)

    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))

    loss = -math.log(qNetwork(batch_state).gather(1, batch_action))

    averagePolicyNetworkOptimizer.zero_grad()
    loss.backward()
    averagePolicyNetworkOptimizer.step()


def learnQNetwork():
    if len(mrl) < BATCH_SIZE:
        return

    transitions = mrl.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))

    current_q_values = qNetwork(batch_state).gather(1, batch_action)
    max_next_q_values = targetPolicyNetwork(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + max_next_q_values
    loss = F.mse_loss(current_q_values, expected_q_values.view(-1, 1))

    qNetworkOptimizer.zero_grad()
    loss.backward()
    qNetworkOptimizer.step()


def RunAgent(game):
    for e in range(EPISODES):

        if random.uniform(0, 1) < ANTICIPATORY_PARAM:
            currentPolicy = qNetwork()
            usingQPolicy = True
        else:
            currentPolicy = averagePolicyNetwork()
            usingQPolicy = False

        state = 123
        action = select_action(currentPolicy, state, usingQPolicy)
        state_next, reward_next = game.take_action(action)

        mrl.push((state, action, reward_next, state_next))

        if usingQPolicy:
            msl.push((state, action))

        learnAveragePolicyNetwork()
        learnQNetwork()

        update_count += 1
        if update_count == TARGET_POLICY_UPDATE_INTERVAL:
            targetPolicyNetwork.load_state_dict(qNetwork.state_dict())
            update_count = 0
