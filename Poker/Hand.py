from .Deck import DeckFactory
from .Player import Player
from typing import List


class LHEHand:
    playerHands = []
    flop = []
    turn = []
    river = []
    pot = 0

    def PreFlop(self):
        for player in self.playersInHand:
            self.playerHands.append(self.deck.pop_cards(2))

        self.playersInHand[self.bigBlindIndex].take_money(self.bigBlind)
        self.playersInHand[self.bigBlindIndex - 1 % len(self.playersInHand)].take_money(self.smallBlind)

    def __init__(self, bigBlind: int, bigBlindIndex: int, playersInHand: List[Player]):
        self.bigBlind = bigBlind
        self.smallBlind = bigBlind / 2
        self.bigBlindIndex = bigBlindIndex
        self.playersInHand = playersInHand
        self.deckFactory = DeckFactory(2)
        self.deck = self.deckFactory.create_deck()

        self.PreFlop()

