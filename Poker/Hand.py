from Deck import Deck
from typing import List
from enum import Enum
import numpy as np
from RL.Agent import Agent
from Player import Player
import torch
from Card import Card
from Score_Detector import HoldemPokerScoreDetector


# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class ActionEncoding():

    def __init__(self, playerIndex: int, round: int, nRaises: int,
                action: int):
        self.playerIndex = playerIndex
        self.round = round
        self.nRaises = nRaises
        self.action = action


class LHEHand:

    def playRound(self, round, nRaises):
        self.round = round
        self.nRaises = nRaises
        self.playerTurn = self.smallBlindIndex #2P

        while not self._roundOver():
            relativePlayerState = Tensor(self.get_relative_player_state())
            self.playersInHand[self.playerTurn].agent.learn(relativePlayerState, 0)
            action = self.playersInHand[self.playerTurn].agent.getAction()[0][0].item()
            self.performAction(action)
            self.playerTurn = 1 - self.playerTurn # rotate turns 2P


    def performAction(self, action):
        if self.nRaises == 4 and action == 2:
            action = 1

        if action == 2:
            betSize = 1 if self.round <= 1 else 2
            self.playersInHand[self.playerTurn].pot = self.playersInHand[1-self.playerTurn].pot + betSize
            self.bettingState[self.playerTurn][self.round][self.nRaises][1] = 1
            self.nRaises += 1

        if action == 1:
            self.playersInHand[self.playerTurn].pot = self.playersInHand[1-self.playerTurn].pot
            self.bettingState[self.playerTurn][self.round][self.nRaises][0] = 1

        if action == 0:
            self.playerOut.append(self.playersInHand.pop(self.playerTurn)) # 2P
            self.winner = 1 - self.playerTurn

        actionEncoding = ActionEncoding(self.playerTurn, self.round, self.nRaises, action)
        self.bettingHistory.append(actionEncoding)


    def _roundOver(self):
        if len(self.playersInHand) == 1:
            return True
        if len(self.bettingHistory) >= 2 and self.bettingHistory[-1].action == 1:
            if self.nRaises == 0 and \
                self.bettingHistory[-1].round == self.round and \
                self.bettingHistory[-2].round == self.round and \
                self.bettingHistory[-2].action == 1:
                return True
            if self.nRaises >= 1:
                return True
        return False


    def get_relative_player_state(self):
        bh = np.vstack((self.bettingState[self.playerTurn:],self.bettingState[:self.playerTurn]))
        ownCards = self.playerHands[self.playerTurn]
        commCards = np.array((self.flop, self.turn, self.river))

        bh = np.ravel(bh)
        ownCards = np.ravel(ownCards)
        commCards = np.ravel(commCards)

        return np.concatenate((bh,ownCards,commCards))


    def __init__(self, smallBlind: int, smallBlindIndex: int, playersInHand: List[Player]):
        #Params
        self.bigBlind = smallBlind*2
        self.smallBlind = smallBlind
        self.smallBlindIndex = smallBlindIndex
        self.playersInHand = playersInHand
        self.playerOut = []

        #Initialize vars
        self.bettingHistory: List[ActionEncoding] = []
        self.bettingState = np.zeros((2,4,5,2)) #2P
        self.playerHands = []
        self._playerHands = [] # Card Object representation
        self._communityCards = [] # Card Object representation
        self.flop = np.zeros(52)
        self.turn = np.zeros(52)
        self.river = np.zeros(52)
        self.playerTurn = self.smallBlindIndex #2P
        self.deck = Deck()

        # Make agents do pre episode tasks
        for player in self.playersInHand:
            player.pot = 0
            player.agent.preEpisodeSetup()

        # PreFlop
        self.playersInHand[self.playerTurn].bet(self.smallBlind)
        self.playersInHand[1 - self.playerTurn].bet(self.bigBlind) #2P
        for player in self.playersInHand:
            playerHand = [0]*52
            playerCards = self.deck.pop_cards(2)
            self._playerHands.append(playerCards)
            for card in playerCards:
                playerHand[card.rank-2+card.suit*13] = 1.0
                self.playerHands.append(playerHand)
        self.playRound(round=0, nRaises=1)

        # Flop
        flopCards = self.deck.pop_cards(3)
        self._communityCards += flopCards
        for card in flopCards:
            self.flop[card.rank-2+card.suit*13] = 1
        self.playRound(round=1, nRaises=0)

        # Turn
        turnCard = self.deck.pop_cards(1)[0]
        self._communityCards += [turnCard]
        self.turn[turnCard.rank-2+turnCard.suit*13] = 1
        self.playRound(round=2, nRaises=0)

        # River
        riverCard = self.deck.pop_cards(1)[0]
        self._communityCards += [riverCard]
        self.river[riverCard.rank-2+riverCard.suit*13] = 1
        self.playRound(round=3, nRaises=0)

        if len(self.playersInHand) == 1: #2P
            amount = self.playerOut[0].pot #2P
            terminalState = Tensor(np.zeros(288))
            self.playerOut[0].agent.learn(terminalState, -amount)
            self.playersInHand[0].agent.learn(terminalState, amount)

        elif len(self.playersInHand) > 1: #2P
            amount = self.playersInHand[0].pot #2P
            terminalState = Tensor(np.zeros(288))
            for player_hand in self._playerHands:
                player_hand += self._communityCards

            score_detector = HoldemPokerScoreDetector()
            p0_score = score_detector.get_score(self._playerHands[0])
            p1_score = score_detector.get_score(self._playerHands[1])
            cmp = p0_score.cmp(p1_score)

            if cmp == 1:
                self.winner = 0
            elif cmp == 0:
                self.winner = -1
            else:
                self.winner = 1
            betSize = self.playersInHand[0].pot
            self.playersInHand[0].agent.learn(terminalState, cmp*betSize)
            self.playersInHand[1].agent.learn(terminalState, -cmp*betSize)
