from collections import deque
import numpy as np

class Player():

    def __init__(self, agent, id):
        self.id = id
        self.agent = agent
        self.total_winnings = 0
        self.last100 = deque(maxlen=100)

    def bet(self, amount):
        amount = min(self.stack_size, amount)
        self.pot += amount
        self.stack_size -= amount
        return amount
    
    def prepare_new_round(self, table_index: int):
        self.hand = [0] * 52
        self.pot = 0
        self.stack_size = max(1, np.random.normal(100, 30))
        self.table_index = table_index
        self.agent.pre_episode_setup()

    def get_action(self, state):
        return self.agent.get_action(state)

    def get_result(self, result: int):
        self.agent.get_result(result)
        self.total_winnings += result
        self.last100.append(result)

        
