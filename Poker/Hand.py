from Deck import Deck
from typing import List
from enum import Enum
import numpy as np
from RL.Agent import Agent
from Player import Player
import torch


# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class ACTION(Enum):
    FOLD = 0
    CALL = 1
    RAISE = 2


class ActionEncoding():
    def __init__(self, playerIndex: int, round: int, nRaises: int,
                action: ACTION):
        self.playerIndex = playerIndex
        self.round = round
        self.nRaises = nRaises
        self.action = action


class LHEHand:

    def playRound(self, round):
        self.round = round
        self.nRaises = 0
        self.playerTurn = self.smallBlindIndex #2P

        while not self._roundOver():
            print("\n\n")
            print(self.get_relative_player_state())
            print("\n\n")
            relativePlayerState = Tensor(self.get_relative_player_state())
            self.playersInHand[self.playerTurn].agent.learn(relativePlayerState, 0)
            action = self.playersInHand[self.playerTurn].agent.getAction()
            self.performAction(action)
            self.playerTurn = 1 - self.playerTurn # rotate turns 2P

        if len(self.playersInHand) == 1: #2P
            amount = self.playerOut[0].pot #2P
            terminalState = Tensor(np.zeros(288))
            self.playerOut[0].agent.learn(terminalState, amount)
            self.playersInHand[0].agent.learn(terminalState, amount)


    def performAction(self, action):
        if self.nRaises == 4 and action == ACTION.RAISE:
            action = ACTION.CALL

        if action == ACTION.RAISE:
            betSize = 1 if self.round <= 1 else 2
            self.playersInHand[self.playerTurn].pot = self.playersInHand[1-self.playerTurn].pot + betSize
            self.bettingState[self.playerTurn][self.round][self.nRaises][1] = 1
            self.nRaises += 1

        if action == ACTION.CALL:
            self.playersInHand[self.playerTurn].pot = self.playersInHand[1-self.playerTurn].pot
            self.bettingState[self.playerTurn][self.round][self.nRaises][0] = 1

        if action == ACTION.FOLD:
            self.playersOut.append(self.playersInHand.pop(self.playerTurn)) # 2P

        actionEncoding = ActionEncoding(self.playerTurn, self.round, self.nRaises, action)
        self.bettingHistory.append(actionEncoding)


    def _roundOver(self):
        if len(self.playersInHand) == 1:
            return True
        if self.nRaises == 4 and self.bettingHistory[-1].action != ACTION.RAISE:  #2P
            return True
        return False


    def get_relative_player_state(self):
        bh = np.vstack((self.bettingState[self.playerTurn:],self.bettingState[:self.playerTurn]))
        ownCards = self.playerHands[self.playerTurn]
        commCards = np.array((self.flop, self.turn, self.river))
        # print(np.ravel(bh))
        # print(np.ravel(ownCards))
        # print(commCards)
        return np.ravel((bh,ownCards,commCards))


    def __init__(self, smallBlind: int, smallBlindIndex: int, playersInHand: List[Player]):
        #Params
        self.bigBlind = smallBlind*2
        self.smallBlind = smallBlind
        self.smallBlindIndex = smallBlindIndex
        self.playersInHand = playersInHand
        self.playerOut = []

        #Initialize vars
        self.bettingHistory: List[ActionEncoding] = []
        self.bettingState = np.zeros((len(playersInHand),4,5,2)) #2P
        self.playerHands = []
        self.flop = np.zeros(52)
        self.turn = np.zeros(52)
        self.river = np.zeros(52)
        self.nRaises = 0
        self.round = 0
        self.playerTurn = 0
        self.deck = Deck()

        # Make agents do pre episode tasks
        for player in self.playersInHand:
            player.agent.preEpisodeSetup()

        # PreFlop
        self.playersInHand[self.playerTurn].bet(self.smallBlind)
        self.playersInHand[1 - self.playerTurn].bet(self.bigBlind) #2P
        for player in self.playersInHand:
            playerHand = [0]*52
            playerCards = self.deck.pop_cards(2)
            for card in playerCards:
                playerHand[card.rank-2+card.suit*13] = 1
                self.playerHands.append(playerHand)
        self.playRound(0)

        # Flop
        flopCards = self.deck.pop_cards(3)
        for card in flopCards:
            self.flop[card.rank-2+card.suit*13] = 1
        self.playRound(1)

        # Turn
        turnCard = self.deck.pop_cards(1)[0]
        self.turn[turnCard.rank-2+turnCard.suit*13] = 1
        self.playRound(2)

        # River
        riverCard = self.deck.pop_cards(1)[0]
        self.river[riverCard.rank-2+riverCard.suit*13] = 1
        self.playRound(3)

        if len(self.playersInHand) > 1: #2P
            wonAmount = self.playerOut[0].pot #2P
            lostAmount = self.playerOut[0].pot #2P
            terminalState = Tensor(np.zeros(288))

            self.playerOut[0].agent.learn(terminalState,0)
            self.playersInHand[0].agent.learn(terminalState,0)
