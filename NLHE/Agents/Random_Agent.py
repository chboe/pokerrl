import random
from Agent import Agent

class Random_Agent(Agent):

    def __init__(self):
        return
    
    def pre_episode_setup(self):
        return

    def get_action(self, state):
        action_value = random.uniform(0, 1)
        return random.randint(0, 1), action_value

    def get_result(self, result: int):
        return 


