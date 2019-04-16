import random

from .move import Move
from .agent import Agent
from .player import Player


class RandomAgent(Agent):
    def __init__(self, plyr: Player):
        super().__init__(plyr)

    def get_next_move(self) -> Move:
        r""" Gets a valid move uniformly at random """
        available_moves = self._plyr.move_set.avail
        values = list(available_moves.values())
        return random.choice(values)
