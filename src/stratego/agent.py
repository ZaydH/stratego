from abc import ABC, abstractmethod

from .move import Move
from .piece import Color
from .player import Player


class Agent(ABC):
    r""" Abstract class encapsulating all Stratego playing engines """
    def __init__(self, player: Player):
        self._plyr = player

    @abstractmethod
    def get_next_move(self) -> Move:
        r""" Agent selects and returns the next move the agent will play """
        pass

    @property
    def color(self) -> Color:
        r""" Return the color of the agent """
        return self._plyr.color
