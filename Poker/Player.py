class Player():


    def bet(self, amount):
        self.pot += amount
        return amount


    def __init__(self, agent):
        self.pot = 0
        self.agent = agent
