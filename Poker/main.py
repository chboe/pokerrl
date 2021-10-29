import random


class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value

    def __str__(self):
        return self.suit + str(self.value)


class Deck:
    _suits = ["Heart", "Diamond", "Spade", "Club"]

    def __init__(self):
        self.deck = []
        for suit in Deck._suits:
            for i in range(1, 14):
                self.deck.append(Card(suit, i))

    def dealCard(self):
        index = random.randint(0, len(self.deck) - 1)
        return self.deck.pop(index)

    def reset(self):
        self.deck = []
        for suit in Deck._suits:
            for i in range(1, 14):
                self.deck.append(Card(suit, i))


class Player:
    def __init__(self, stackSize):
        self.stack = stackSize
        self.hand = []

    def getCard(self, card):
        self.hand.append(card)

    def bet(self, amount):
        amountToBet = min(amount, self.stack)
        self.stack = self.stack - amountToBet
        return amountToBet


class Table:
    def __init__(self, smallBlind=None, bigBlind=None, playerCount=6, startStack=1000):
        self.players = []
        for i in range(playerCount):
            self.players.append(Player(startStack))
        self.bigBlindIndex = 1
        self.smallBlindIndex = 0

        self.smallBlind = startStack / 100
        if (smallBlind != None):
            self.smallBlind = smallBlind

        self.bigBlind = startStack / 50
        if (bigBlind != None):
            self.bigBlind = bigBlind

        self.deck = Deck()


class Round:
    def __init__(self, table):
        self.pot = 0
        self.pot += table.players[table.bigBlindIndex].bet(table.bigBlind)
        self.pot += table.players[table.smallBlindIndex].bet(table.smallBlind)

        playersInRound = []
        for player in table.players:
            playersInRound.append(player)

        for player in table.players:
            player.getCard(table.deck.dealCard())
            player.getCard(table.deck.dealCard())

        startPlayer = playersInRound[(table.bigBlindIndex + 1) % len(playersInRound)]

        actions = []
        actions[-(playersInRound-1):]