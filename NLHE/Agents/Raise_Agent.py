import random
from Agent import Agent

class Raise_Agent(Agent):

    def __init__(self, **kwargs):
        if 'id' in kwargs:
            self.id = kwargs['id']
        return
    
    def pre_episode_setup(self):
        return

    def get_action(self, state):
        return 0

    def get_result(self, result: int):
        return 