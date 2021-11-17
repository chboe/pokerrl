import math
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

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

steps_done = 0
update_count = 0

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(2, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 1024)
        self.l4 = nn.Linear(1024, 512)
        self.l5 = nn.Linear(512, 3)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)
        return x


def select_action(currentPolicy, state, usingQpolicy):
    global steps_done
    CURR_EPS = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if random.uniform(0, 1) > CURR_EPS or usingQpolicy:
        return currentPolicy(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(3)]])


def RunAgent(game):
    global update_count
    mrl = ReplayMemory(MRL_SIZE)
    msl = ReplayMemory(MSL_SIZE)
    averagePolicyNetwork = Network()
    qNetwork = Network()
    targetPolicyNetwork = Network()
    targetPolicyNetwork.load_state_dict(qNetwork.state_dict())
    for e in range(EPISODES):

        if (random.uniform(0, 1) < ANTICIPATORY_PARAM):
            currentPolicy = qNetwork()
            usingQPolicy = True
        else:
            currentPolicy = averagePolicyNetwork()
            usingQPolicy = False

        state = 123
        action = select_action(currentPolicy, state, usingQPolicy)
        state_next, reward_next = game.take_action(action)

        mrl.push((state, action, reward_next, state_next))

        if (usingQPolicy):
            msl.push((state, action))

        # TODO POLICY GRADIENT

        update_count += 1
        if update_count == TARGET_POLICY_UPDATE_INTERVAL:
            targetPolicyNetwork.load_state_dict(qNetwork.state_dict())
            update_count = 0
