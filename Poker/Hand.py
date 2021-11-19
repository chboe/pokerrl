from Deck import DeckFactory
from Player import Player
from typing import List

class LHEHand:


    def PreFlop(self):
        for player in self.playersInHand:
            self.playerHands.append(self.deck.pop_cards(2))

        self.playersInHand[self.bigBlindIndex].take_money(self.bigBlind)
        self.playersInHand[self.bigBlindIndex - 1 % len(self.playersInHand)].take_money(self.smallBlind)
        while (len(
                    filter(
                        lambda playerIndex, round, numberOfRaises, action :
                        round == self.round and (action == 0 or action == 1),
                        self.bettingHistory
                    )
                ) > (len(self.playersInHand)-1)):





    def __init__(self, bigBlind: int, bigBlindIndex: int, playersInHand: List[Player]):
        #Params
        self.bigBlind = bigBlind
        self.smallBlind = bigBlind / 2
        self.bigBlindIndex = bigBlindIndex
        self.playersInHand = playersInHand

        #Initialize vars
        self.bettingHistory = []  # (playerIndex, round, numberOfRaises, action)
        self.playerHands = []
        self.flop = []
        self.turn = []
        self.river = []
        self.round = 0
        self.pot = 0
        self.deckFactory = DeckFactory(2)
        self.deck = self.deckFactory.create_deck()

        self.PreFlop()