class Game:
    players = []

    def playerJoin(self, player):
        self.players.append(player)

    def playerLeave(self, player):
        self.players.remove(player)

