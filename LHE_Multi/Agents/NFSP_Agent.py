import random
from Agent import Agent
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import re
from operator import itemgetter


# if gpu is to be used (LARGE NETWORKS ONLY)
#use_cuda = torch.cuda.is_available() 
use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
torch.set_default_tensor_type(Tensor)

_TERMINAL_STATE = torch.zeros(748)[None, :]


class ExponentialReservoir():
    def __init__(self, capacity):
        self.stream_age = 0
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if self.stream_age < self.capacity: 
            self.memory.append(transition)
        elif random.uniform(0, 1) < 0.3:
            eject_index = random.randint(0, self.capacity - 1)
            self.memory[eject_index] = transition
        self.stream_age += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class CircularBuffer():
    def __init__(self, capacity):
        self.memory = [0]*capacity # MALLOC capacity size
        self.capacity = capacity
        self.pointer = 0
        self.length = 0

    def push(self, transition):
        if self.length < self.capacity:
            self.memory[self.length] = transition
            self.length += 1
        else:
            self.memory[self.pointer] = transition
            self.pointer = (self.pointer + 1) % self.capacity
    
    def sample(self, batch_size):
        indices = random.sample(range(self.length), batch_size)
        return itemgetter(*indices)(self.memory)
    
    def __len__(self):
        return self.length


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(288, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64, 3)

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
                 EPS: float,
                 EPS_DECAY: int,
                 MODEL_TO_LOAD: str = None,
                 LEARN: bool = True):

        self.id = id
        self.SAVE_INTERVAL = SAVE_INTERVAL
        self.MRL_SIZE = MRL_SIZE
        self.MSL_SIZE = MSL_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.TARGET_POLICY_UPDATE_INTERVAL = TARGET_POLICY_UPDATE_INTERVAL
        self.ANTICIPATORY_PARAM = ANTICIPATORY_PARAM
        self.EPS = EPS
        self.EPS_DECAY = EPS_DECAY
        self.LEARN = LEARN

        # Initializing Memories
        self.mrl = CircularBuffer(self.MRL_SIZE)
        self.msl = ExponentialReservoir(self.MSL_SIZE)

        # Initialize Q, Q' and Avg networks.
        self.targetPolicyNetwork = Network()
        if MODEL_TO_LOAD != None:
            self.qNetwork = torch.load(MODEL_TO_LOAD + "_target.model")
            self.averagePolicyNetwork = torch.load(MODEL_TO_LOAD + "_avg.model")
            self.update_count = int(re.findall(r'steps=(\d*)', MODEL_TO_LOAD)[-1])
        else:
            self.qNetwork = Network()
            self.averagePolicyNetwork = Network()
            self.update_count = 0

        # Make networks use cuda
        if use_cuda:
            self.qNetwork.cuda()
            self.targetPolicyNetwork.cuda()
            self.averagePolicyNetwork.cuda()

        self.targetPolicyNetwork.load_state_dict(self.qNetwork.state_dict())
        self.qNetworkOptimizer = optim.Adam(self.qNetwork.parameters(), RL_LR)
        self.averagePolicyNetworkOptimizer = optim.Adam(self.averagePolicyNetwork.parameters(), SL_LR)


    def learnAveragePolicyNetwork(self):
        if len(self.msl) < self.BATCH_SIZE:
            return

        transitions = self.msl.sample(self.BATCH_SIZE)
        
        batch_state, batch_action = zip(*transitions)
        batch_state = torch.cat(batch_state)
        batch_action = torch.cat(batch_action)

        batch_pred = self.averagePolicyNetwork(batch_state)
        loss = nn.CrossEntropyLoss()(batch_pred, batch_action.flatten())
        
        self.averagePolicyNetworkOptimizer.zero_grad()
        loss.backward()
        self.averagePolicyNetworkOptimizer.step()
        

    def learnQNetwork(self):
        if len(self.mrl) < self.BATCH_SIZE:
            return

        transitions = self.mrl.sample(self.BATCH_SIZE)
        batch_state, batch_action, batch_next_state, batch_reward, batch_not_terminal = zip(*transitions)

        batch_state = torch.cat(batch_state)
        batch_action = torch.cat(batch_action)
        batch_reward = torch.cat(batch_reward)
        batch_next_state = torch.cat(batch_next_state)
        batch_not_terminal = torch.cat(batch_not_terminal)
        
        current_q_values = self.qNetwork(batch_state).gather(1, batch_action)
        max_next_q_values = self.targetPolicyNetwork(batch_next_state).detach().max(1)[0]
        max_next_q_values *= batch_not_terminal
        expected_q_values = batch_reward + max_next_q_values

        loss = F.mse_loss(current_q_values, expected_q_values.view(-1, 1))
        self.qNetworkOptimizer.zero_grad()
        loss.backward()
        self.qNetworkOptimizer.step()
        

    def update_state(self, next_state, next_reward: int, not_terminal: int):
        if self.state == None: 
            self.state = next_state
            return

        if self.LEARN:
            if self.currentPolicy == self.qNetwork:
                self.msl.push((self.state, self.action))
            self.mrl.push((self.state, self.action, next_state, Tensor([next_reward]), Tensor([not_terminal])))
            self.learnAveragePolicyNetwork()
            self.learnQNetwork()

            self.update_count += 1
            if self.update_count % self.TARGET_POLICY_UPDATE_INTERVAL == 0:
                self.targetPolicyNetwork.load_state_dict(self.qNetwork.state_dict())
            if self.update_count % self.SAVE_INTERVAL == 0:
                torch.save(self.targetPolicyNetwork, f'Agents/NFSP_Model/id={self.id}_steps={self.update_count}_target.model')
                torch.save(self.averagePolicyNetwork, f'Agents/NFSP_Model/id={self.id}_steps={self.update_count}_avg.model')

        self.state = next_state

    def select_action(self, state):
        self.update_state(state, 0, 1)

        if self.currentPolicy == self.qNetwork:
            if random.uniform(0, 1) > (self.EPS - self.EPS * min(1, self.update_count/self.EPS_DECAY)):
                pred = self.currentPolicy(state)
                self.action = pred.data.max(1)[1].view(1, 1)
            else:
                self.action = LongTensor([[random.randrange(3)]])
        else:
            pred = self.currentPolicy(state)
            pred = F.softmax(pred, dim=1).data[0]
            prob = random.uniform(0,1)

            for action in range(len(pred)):
                prob -= pred[action]
                if prob < 0:
                    self.action = LongTensor([[action]])
                    return self.action[0][0].item()

    def pre_episode_setup(self):
        self.state = None
        # During evaluation (self.LEARN=False), the avg policy network is chosen always
        if random.uniform(0, 1) < self.ANTICIPATORY_PARAM:
            self.currentPolicy = self.qNetwork
        else:
            self.currentPolicy = self.averagePolicyNetwork
        self.currentPolicy = self.averagePolicyNetwork # TODO remove this for training

    def get_action(self, state):
        return self.select_action(Tensor(state)[None, :])

    def get_result(self, result: int):
        self.update_state(_TERMINAL_STATE, result, 0)

