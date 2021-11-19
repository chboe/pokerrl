import random
from typing import List
from Card import Card

class Deck:
    def __init__(self):
        self._cards: List[Card] = [Card(rank, suit) for rank in range(2,15) for suit in range(4)]
        self._discard: List[Card] = []
        random.shuffle(self._cards)

    def pop_cards(self, num_cards=1) -> List[Card]:
        """Returns and removes cards them from the top of the deck."""
        new_cards = []
        if len(self._cards) < num_cards:
            new_cards = self._cards
            self._cards = self._discard
            self._discard = []
            random.shuffle(self._cards)
        return new_cards + [self._cards.pop() for _ in range(num_cards - len(new_cards))]

    def push_cards(self, discard: List[Card]):
        """Adds discard"""
        self._discard += discard
