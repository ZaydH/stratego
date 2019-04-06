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
    r""" Encapsulates a player """
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

    _all = []

    def __init__(self, rank: Union[str, int]):
        assert not isinstance(rank, str) or rank in {Rank.SPY, Rank.BOMB, Rank.FLAG}
        assert not isinstance(rank, int) or Rank.MIN() <= rank <= Rank.MAX()
        self._val = rank

    @staticmethod
    def get_all() -> List['Rank']:
        r""" Generate a list of all valid ranks"""
        if not Rank._all:
            all_rank = list(range(Rank.MIN(), Rank.MAX() + 1)) + [Rank.SPY, Rank.BOMB, Rank.FLAG]
            Rank._all = [Rank(r) for r in all_rank]
        return Rank._all

    @property
    def value(self) -> Union[str, int]:
        r""" Accessor for the rank's value """
        return self._val

    @staticmethod
    def MIN() -> int: return 1  # pylint: disable=missing-docstring, invalid-name

    @staticmethod
    def MAX() -> int: return 9  # pylint: disable=missing-docstring,invalid-name

    def is_immobile(self) -> bool:
        r"""
        Checks if the rank allows the piece to move.
        >>> f = Rank(Rank.FLAG); b = Rank(Rank.BOMB); s= Rank(Rank.SPY)
        >>> print(f.is_immobile(), b.is_immobile(), s.is_immobile())
        True True False
        """
        return self._val == Rank.FLAG or self._val == Rank.BOMB

    def __eq__(self, other) -> bool:
        r"""
        Returns True if the two ranks are equal

        >>> p1 = Rank(Rank.MARSHALL); b = Rank(Rank.BOMB); s = Rank(Rank.SPY)
        >>> print(p1 == p1, p1 == b, s == p1)
        True False False
        """
        assert self.value != Rank.FLAG and self.value != Rank.BOMB, "Attacker cant be stationary"
        return self.value == other.value

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

    def __hash__(self):
        return hash(str(self.value))

    def board_file_str(self) -> str:
        r""" Identifier for rank entries in the board file """
        return "Rank_" + str(self)

    @staticmethod
    def get_from_board_file(line_header: str) -> 'Rank':
        r""" Convert a line header from a board file to a corresponding Rank object. """
        spl = line_header.split("_", max=1)
        try:
            return Rank(int(spl[1]))
        except ValueError:
            return Rank(spl[1])


class Piece:
    r""" Per piece analysis """
    def __init__(self, color: Player, rank):
        self._color = color
        self._rank = rank if isinstance(rank, Rank) else Rank(rank)

    @property
    def rank(self) -> Rank: return self._rank  # pylint: disable=missing-docstring, invalid-name

    @property
    def color(self) -> Player: return self._color  # pylint: disable=missing-docstring, invalid-name

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
