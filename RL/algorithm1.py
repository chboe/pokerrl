import math
import random
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.autograd import Variable
from RL.types import ReplayMemory, Network

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
        self.prevState = DEFAULT_STATE

        # if gpu is to be used
        use_cuda = torch.cuda.is_available()
        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
        self.Tensor = self.FloatTensor

    def select_action(self, currentPolicy, state, usingQpolicy):
        if random.uniform(0, 1) > self.EPS or usingQpolicy:
            return currentPolicy(Variable(state).type(self.FloatTensor)).data.max(1)[1].view(1, 1)
        else:
            return self.LongTensor([[random.randrange(3)]])

    def learnAveragePolicyNetwork(self):
        if len(self.msl) < self.BATCH_SIZE:
            return

        transitions = self.msl.sample(self.BATCH_SIZE)
        batch_state, batch_action = zip(*transitions)

        batch_state = Variable(torch.cat(batch_state))
        batch_action = Variable(torch.cat(batch_action))

        loss = -math.log(self.qNetwork(batch_state).gather(1, batch_action))

        self.averagePolicyNetworkOptimizer.zero_grad()
        loss.backward()
        self.averagePolicyNetworkOptimizer.step()

    def learnQNetwork(self):
        if len(self.mrl) < self.BATCH_SIZE:
            return

        transitions = self.mrl.sample(self.BATCH_SIZE)
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

        batch_state = Variable(torch.cat(batch_state))
        batch_action = Variable(torch.cat(batch_action))
        batch_reward = Variable(torch.cat(batch_reward))
        batch_next_state = Variable(torch.cat(batch_next_state))

        current_q_values = self.qNetwork(batch_state).gather(1, batch_action)
        max_next_q_values = self.targetPolicyNetwork(batch_next_state).detach().max(1)[0]
        expected_q_values = batch_reward + max_next_q_values
        loss = F.mse_loss(current_q_values, expected_q_values.view(-1, 1))

        self.qNetworkOptimizer.zero_grad()
        loss.backward()
        self.qNetworkOptimizer.step()

    def GetAction(self, state):
        if random.uniform(0, 1) < self.ANTICIPATORY_PARAM:
            currentPolicy = self.qNetwork()
            usingQPolicy = True
        else:
            currentPolicy = self.averagePolicyNetwork()
            usingQPolicy = False

        action = self.select_action(currentPolicy, state, usingQPolicy)
        state_next, reward_next, = game.take_action(action)

        self.mrl.push((state, action, reward_next, state_next))

        if usingQPolicy:
            self.msl.push((state, action))

        self.learnAveragePolicyNetwork()
        self.learnQNetwork()

        self.update_count += 1
        if self.update_count == self.TARGET_POLICY_UPDATE_INTERVAL:
            self.targetPolicyNetwork.load_state_dict(self.qNetwork.state_dict())
            self.update_count = 0

        self.prevState = state
        return action