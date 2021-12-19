from collections import deque
import time
from Card import Card
actions = ['raises', 'calls', 'folds']


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


class Player_Debug():

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
        print(f'{self.agent.id} sits at {table_index}')
        self.agent.pre_episode_setup()

    def get_action(self, state):
        print(f'{self.agent.id}')
        print(state[:40])
        print(state[40:80])
        print(state[80:132])
        action = self.agent.get_action(state)
        print(f'{self.agent.id} {actions[action]}')
        return action

    def get_result(self, result: int):
        self.agent.get_result(result)
        self.total_winnings += result
        print(f'{self.agent.id} received {result}')
        player_cards = Card.decode_card_array(self.hand)
        for c in player_cards:
            print(Card.RANKS[c.rank], Card.SUITS[c.suit])
        self.last100.append(result)

        
