import math
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.autograd import Variable
from RL.Types import ReplayMemory, Network
import numpy as np


# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class Agent:

    def __init__(self,
                 MRL_SIZE: int,
                 MSL_SIZE: int,
                 RL_LR: float,
                 SL_LR: float,
                 BATCH_SIZE: int,
                 TARGET_POLICY_UPDATE_INTERVAL: int,
                 ANTICIPATORY_PARAM: float,
                 EPS: float):

        self.MRL_SIZE = MRL_SIZE
        self.MSL_SIZE = MSL_SIZE
        self.RL_LR = RL_LR
        self.SL_LR = SL_LR
        self.BATCH_SIZE = BATCH_SIZE
        self.TARGET_POLICY_UPDATE_INTERVAL = TARGET_POLICY_UPDATE_INTERVAL
        self.ANTICIPATORY_PARAM = ANTICIPATORY_PARAM
        self.EPS = EPS

        # Initializing everything
        self.update_count = 0
        self.mrl = ReplayMemory(self.MRL_SIZE)
        self.msl = ReplayMemory(self.MSL_SIZE)
        self.averagePolicyNetwork = Network()
        self.qNetwork = Network()
        self.targetPolicyNetwork = Network()
        self.targetPolicyNetwork.load_state_dict(self.qNetwork.state_dict())
        self.qNetworkOptimizer = optim.SGD(self.qNetwork.parameters(), self.RL_LR)
        self.averagePolicyNetworkOptimizer = optim.SGD(self.averagePolicyNetwork.parameters(), self.SL_LR)


    def select_action(self, state):
        if self.currentPolicy == self.qNetwork:
            if random.uniform(0, 1) > self.EPS:
                pred = self.currentPolicy(Variable(state).type(FloatTensor))
                return pred.data.max(1)[1].view(1, 1)
            else:
                return LongTensor([[random.randrange(3)]])
        else:
            pred = self.currentPolicy(Variable(state).type(FloatTensor))
            pred = F.softmax(pred, dim=1).data[0]
            prob = random.uniform(0,1)

            for action in range(len(pred)):
                prob -= pred[action]
                if prob < 0:
                    return LongTensor([[action]])

    def learnAveragePolicyNetwork(self):
        print(len(self.msl))
        if len(self.msl) < self.BATCH_SIZE:
            return

        transitions = self.msl.sample(self.BATCH_SIZE)
        batch_state, batch_action = zip(*transitions)

        batch_state = Variable(torch.cat(batch_state))
        batch_action = Variable(torch.cat(batch_action))

        batch_pred = self.averagePolicyNetwork(batch_state)
        loss = nn.CrossEntropyLoss()(batch_pred, batch_action.data.flatten())

        self.averagePolicyNetworkOptimizer.zero_grad()
        loss.backward()
        self.averagePolicyNetworkOptimizer.step()

    def learnQNetwork(self):
        if len(self.mrl) < self.BATCH_SIZE:
            return

        transitions = self.mrl.sample(self.BATCH_SIZE)
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

        batch_state = torch.cat(batch_state)
        batch_action = torch.cat(batch_action)
        batch_reward = torch.cat(batch_reward)
        batch_next_state = torch.cat(batch_next_state)

        # Make vector of 0 and 1s based on terminal states
        terminalStates = list(map(lambda x: int(x.sum().item() != 0), batch_next_state))

        current_q_values = self.qNetwork(batch_state).gather(1, batch_action)
        max_next_q_values = self.targetPolicyNetwork(batch_next_state).detach().max(1)[0]
        max_next_q_values *= Tensor(terminalStates)
        expected_q_values = batch_reward + max_next_q_values

        loss = F.mse_loss(current_q_values, expected_q_values.view(-1, 1))

        self.qNetworkOptimizer.zero_grad()
        loss.backward()
        self.qNetworkOptimizer.step()


    def preEpisodeSetup(self):
        self.state = None
        if random.uniform(0, 1) < self.ANTICIPATORY_PARAM:
            self.currentPolicy = self.qNetwork
        else:
            self.currentPolicy = self.averagePolicyNetwork

    def getAction(self):
        self.action = self.select_action(FloatTensor(self.state)[None, :])
        return self.action

    def learn(self, next_state, next_reward):
        if self.state == None:
            self.state = next_state
            return

        self.mrl.push((self.state[None, :], self.action, next_state[None, :], FloatTensor([next_reward])))

        if self.currentPolicy == self.qNetwork:
            self.msl.push((self.state[None, :], self.action))

        self.learnAveragePolicyNetwork()
        self.learnQNetwork()

        self.update_count += 1
        if self.update_count == self.TARGET_POLICY_UPDATE_INTERVAL:
            self.targetPolicyNetwork.load_state_dict(self.qNetwork.state_dict())
            self.update_count = 0

        self.state = next_state