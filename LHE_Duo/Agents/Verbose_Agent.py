import random
from Agent import Agent

class Verbose_Agent(Agent):

    def __init__(self, id):
        self.id = id
        return
    
    def pre_episode_setup(self):
        return

    def get_action(self, state):
        action = random.randint(0, 2)
        return action

    def get_result(self, result: int):
        return 