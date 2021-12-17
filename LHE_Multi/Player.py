from collections import deque

class Player():

    def __init__(self, agent, id):
        self.id = id
        self.agent = agent
        self.total_winnings = 0
        self.last100 = deque(maxlen=100)

    def bet(self, amount):
        self.pot += amount
        return amount
    
    def prepare_new_round(self, table_index: int):
        self.hand = [0] * 52
        self.pot = 0
        self.table_index = table_index
        self.agent.pre_episode_setup()

    def get_action(self, state):
        return self.agent.get_action(state)

    def get_result(self, result: int):
        self.agent.get_result(result)
        self.total_winnings += result
        self.last100.append(result)

        
