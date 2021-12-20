from collections import deque
import numpy as np

class Player():

    def __init__(self, agent, id):
        self.id = id
        self.agent = agent
        self.total_winnings = 0
        self.last100 = deque(maxlen=100)

    def bet(self, amount: float) -> float:
        current_stack_size = self.current_stack_size() # How much does player have
        amount = min(current_stack_size, amount) # How much can he bet
        self.current_stack_pct = (current_stack_size - amount) / self.start_stack_size
        self.pot += amount
        return amount
    
    def prepare_new_round(self, table_index: int):
        self.hand = [0] * 52
        self.pot = 0
        self.start_stack_size = max(10, np.random.normal(100, 30))
        self.current_stack_pct = 1.0 # 100%
        self.table_index = table_index
        self.agent.pre_episode_setup()

    def current_stack_size(self):
        return self.current_stack_pct * self.start_stack_size

    def get_action(self, state):
        return self.agent.get_action(state)

    def get_result(self, result: int):
        self.agent.get_result(result)
        self.total_winnings += result
        self.last100.append(result)

        
