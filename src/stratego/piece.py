# -*- coding: utf-8 -*-
r"""
    stratego.piece
    ~~~~~~~~~~~~~~

    Encapsulates the \p Rank and \p Pieces classes

    :copyright: (c) 2019 by Steven Walton & Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""

from enum import Enum
from typing import Union, List


class Player(Enum):
    Red = 0
    Blue = 1


class Rank:
    r"""
    Standardizes handling of piece rank.  Could optionally be upgraded to both European and
    American ranking style.
    """
    SPY = 'S'
    BOMB = 'B'
    FLAG = 'F'

    MARSHALL = 1
    MINER = 8

    def __init__(self, rank: Union[str, int]):
        assert not isinstance(rank, str) or rank in {Rank.SPY, Rank.BOMB, Rank.FLAG}
        assert not isinstance(rank, int) or Rank.MIN() <= rank <= Rank.MAX()
        self._val = rank

    @staticmethod
    def all() -> List[Union[int, str]]:
        r""" Generate a list of all valid ranks"""
        return list(range(Rank.MIN(), Rank.MAX() + 1)) + [Rank.SPY, Rank.BOMB, Rank.FLAG]

    @property
    def value(self) -> Union[str, int]:
        r""" Accessor for the rank's value """
        return self._val

    @staticmethod
    def MIN() -> int: return 1

    @staticmethod
    def MAX() -> int: return 9

    def is_immobile(self) -> bool:
        r"""
        Checks if the rank allows the piece to move.
        >>> f = Rank(Rank.FLAG); b = Rank(Rank.BOMB); s= Rank(Rank.SPY)
        >>> print(f.is_immobile(), b.is_immobile(), s.is_immobile())
        True True False
        """
        return self._val == Rank.FLAG or self._val == Rank.BOMB

    def __eq__(self, other) -> bool:
        assert self.value != Rank.FLAG and self.value != Rank.BOMB, "Attacker cant be stationary"
        self.value == other.value

    def __gt__(self, other: 'Rank') -> bool:
        r"""
        Checks if the implicit rank defeats the \p other rank.

        :param other: Defending rank
        :return: True if the attacking (left) rank beats the defending (right) rank

        >>> p1, p8 = Rank(1), Rank(8)
        >>> s = Rank(Rank.SPY); b = Rank(Rank.BOMB); f = Rank(Rank.FLAG)
        >>> print(p1 > s, s > p1, s > p8, p8 > s, s > f, p8 > f)
        True True False True True True
        >>> print(p8 > b, p1 > b)
        True False
        """
        assert self.value != Rank.FLAG and self.value != Rank.BOMB, "Attacker cant be stationry"
        if other.value == Rank.FLAG:
            return True
        if self.value == Rank.SPY:
            return other.value == self.MARSHALL
        if other.value == Rank.SPY:
            return True
        if other.value == Rank.BOMB:
            return self.value == self.MINER
        return self.value < other.value

    def __str__(self) -> str:
        return str(self.value)


class Piece:
    def __init__(self, color: Player, rank):
        self._color = color
        self._rank = rank if isinstance(rank, Rank) else Rank(rank)

    @property
    def rank(self) -> Rank: return self._rank

    @property
    def color(self) -> Player: return self._color

    def is_immobile(self) -> True:
        r""" Returns True if piece cannot move """
        return self._rank.is_immobile()

    def __gt__(self, other: 'Piece') -> bool:
        assert self.color != other.color, "Player cannot attack itself"
        return self.rank > other.rank

    def __eq__(self, other: 'Piece') -> bool:
        assert self.color != other.color, "Player cannot attack itself"
        return self.rank == other.rank


if __name__ == "__main__":
    import doctest
    doctest.testmod()
