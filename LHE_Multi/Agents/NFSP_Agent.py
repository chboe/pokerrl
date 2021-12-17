import random
from Agent import Agent
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


# if gpu is to be used
use_cuda = False
#use_cuda = torch.cuda.is_available()
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

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(748, 1024)
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


class NFSP_Agent(Agent):

    def __init__(self,
                 id: int,
                 SAVE_INTERVAL: int,
                 MRL_SIZE: int,
                 MSL_SIZE: int,
                 RL_LR: float,
                 SL_LR: float,
                 BATCH_SIZE: int,
                 TARGET_POLICY_UPDATE_INTERVAL: int,
                 ANTICIPATORY_PARAM: float,
                 EPS: float):

        self.id = id
        self.SAVE_INTERVAL = SAVE_INTERVAL
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


    def select_action(self, state, reward):
        self.update_state(state, reward)

        if self.currentPolicy == self.qNetwork:
            if random.uniform(0, 1) > self.EPS:
                pred = self.currentPolicy(Variable(self.state[None, :]).type(FloatTensor))
                self.action = pred.data.max(1)[1].view(1, 1)
            else:
                self.action = LongTensor([[random.randrange(3)]])
        else:
            pred = self.currentPolicy(Variable(self.state[None, :]).type(FloatTensor))
            pred = F.softmax(pred, dim=1).data[0]
            prob = random.uniform(0,1)

            for action in range(len(pred)):
                prob -= pred[action]
                if prob < 0:
                    self.action = LongTensor([[action]])
        return self.action[0][0].item()

    def learnAveragePolicyNetwork(self):
        if len(self.msl) < self.BATCH_SIZE:
            return

        transitions = self.msl.sample(self.BATCH_SIZE)
        batch_state, batch_action = zip(*transitions)

        batch_state = Variable(torch.cat(batch_state))
        batch_action = Variable(torch.cat(batch_action))

        batch_pred = self.averagePolicyNetwork(batch_state)
        loss = nn.CrossEntropyLoss()(batch_pred, batch_action.flatten())

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
        if current_q_values.isnan().any(): # DEBUG
            print(current_q_values) # DEBUG
        max_next_q_values = self.targetPolicyNetwork(batch_next_state).detach().max(1)[0]
        max_next_q_values *= Tensor(terminalStates)
        expected_q_values = batch_reward + max_next_q_values

        loss = F.mse_loss(current_q_values, expected_q_values.view(-1, 1))

        self.qNetworkOptimizer.zero_grad()
        loss.backward()
        self.qNetworkOptimizer.step()

    def update_state(self, next_state, next_reward):
        # Decide whether to learn
        if self.state == None: 
            self.state = next_state
            return

        self.mrl.push((self.state[None, :], self.action, next_state[None, :], FloatTensor([next_reward])))

        if self.currentPolicy == self.qNetwork:
            self.msl.push((self.state[None, :], self.action))
        
        self.learnAveragePolicyNetwork()
        self.learnQNetwork()
        self.state = next_state

        self.update_count += 1
        if self.update_count % self.TARGET_POLICY_UPDATE_INTERVAL == 0:
            self.targetPolicyNetwork.load_state_dict(self.qNetwork.state_dict())
        
        if self.update_count % self.SAVE_INTERVAL == 0:
            torch.save(self.targetPolicyNetwork, f'id={self.id}_steps={self.update_count}')

    def pre_episode_setup(self):
        self.state = None
        if random.uniform(0, 1) < self.ANTICIPATORY_PARAM:
            self.currentPolicy = self.qNetwork
        else:
            self.currentPolicy = self.averagePolicyNetwork

    def get_action(self, state):
        return self.select_action(FloatTensor(state), 0)

    def get_result(self, result: int):
        self.update_state(FloatTensor(np.zeros(748)), result)

