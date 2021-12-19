import random
from Agent import Agent

class Random_Agent(Agent):

    def __init__(self):
        return
    
    def pre_episode_setup(self):
        return

    def get_action(self, state):
        return random.randint(0, 2)

    def get_result(self, result: int):
        return 


