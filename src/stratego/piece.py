# -*- coding: utf-8 -*-
r"""
    stratego.piece
    ~~~~~~~~~~~~~~

    Implementation of the \p Rank and \p Pieces classes

    :copyright: (c) 2019 by Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""

from enum import Enum
from typing import Union, List

from .location import Location


class Color(Enum):
    r""" Encapsulates a player """
    RED = 0
    BLUE = 1

    def get_next(self):
        r""" Get the next turn's player color.
        >>> print(Color.RED.get_next())
        Color.BLUE
        >>> print(Color.BLUE.get_next())
        Color.RED
        """
        return Color.RED if self == Color.BLUE else Color.BLUE


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
    SCOUT = 9

    _static_pieces = dict()
    _STR_PIECES = {SPY, BOMB, FLAG}
    _all = []

    def __init__(self, rank: Union[str, int]):
        if isinstance(rank, str):
            if rank not in Rank._STR_PIECES:
                rank = int(rank)
        assert not isinstance(rank, int) or Rank.MIN() <= rank <= Rank.MAX()
        self._val = rank

    @staticmethod
    def get_all() -> List['Rank']:
        r""" Generate a list of all valid ranks"""
        if not Rank._all:
            all_rank = list(range(Rank.MIN(), Rank.MAX() + 1)) + [Rank.SPY, Rank.BOMB, Rank.FLAG]
            Rank._all = [Rank(r) for r in all_rank]
        return Rank._all

    @staticmethod
    def count() -> int:
        r""" Accessor for the total number of ranks """
        return len(Rank.get_all())

    @staticmethod
    def moveable_count() -> int:
        return Rank.count() - 2  # Exclude bomb and flag

    @property
    def value(self) -> Union[str, int]:
        r""" Accessor for the rank's value """
        return self._val

    # noinspection PyPep8Naming
    @staticmethod
    def MIN() -> int: return 1  # pylint: disable=missing-docstring, invalid-name

    # noinspection PyPep8Naming
    @staticmethod
    def MAX() -> int: return 9  # pylint: disable=missing-docstring,invalid-name

    @staticmethod
    def marshall() -> 'Rank':
        r""" Accessor for the "Marshall" rank """
        return Rank._static_piece(Rank.MARSHALL)

    @staticmethod
    def miner() -> 'Rank':
        r""" Accessor for the "Miner" rank """
        return Rank._static_piece(Rank.MINER)

    @staticmethod
    def scout() -> 'Rank':
        r""" Accessor for the "Scout" rank """
        return Rank._static_piece(Rank.SCOUT)

    @staticmethod
    def spy() -> 'Rank':
        r""" Accessor for the "Spy" rank """
        return Rank._static_piece(Rank.SPY)

    @staticmethod
    def flag() -> 'Rank':
        r""" Accessor for the "Flag" rank """
        return Rank._static_piece(Rank.FLAG)

    @staticmethod
    def bomb() -> 'Rank':
        r""" Accessor for the "Bomb" rank """
        return Rank._static_piece(Rank.BOMB)

    @staticmethod
    def _static_piece(r: Union[str, int]):
        r""" Extracts a static piece, adding it to the static piece dictionary if necessary """
        try: return Rank._static_pieces[r]
        except KeyError:
            Rank._static_pieces[r] = Rank(r)
            return Rank._static_pieces[r]

    def is_immobile(self) -> bool:
        r"""
        Checks if the rank allows the piece to move.
        >>> f = Rank.flag(); b = Rank.bomb(); s = Rank.spy()
        >>> print(f.is_immobile(), b.is_immobile(), s.is_immobile())
        True True False
        """
        return self._val == Rank.FLAG or self._val == Rank.BOMB

    def __eq__(self, other) -> bool:
        r"""
        Returns True if the two ranks are equal

        >>> p1 = Rank.marshall(); b = Rank.bomb(); s = Rank.spy()
        >>> print(p1 == p1, p1 == b, s == p1)
        True False False
        """
        # Removed this check since it would affect searching in sets/dictionaries
        # assert self.value != Rank.FLAG and self.value != Rank.BOMB, "Attacker cant be stationary"
        return self.value == other.value

    def __gt__(self, other: 'Rank') -> bool:
        r"""
        Checks if the implicit rank defeats the \p other rank.

        :param other: Defending rank
        :return: True if the attacking (left) rank beats the defending (right) rank

        >>> p1, p8 = Rank(1), Rank(8)
        >>> s = Rank.spy(); b = Rank.bomb(); f = Rank.flag()
        >>> print(p1 > s, s > p1, s > p8, p8 > s, s > f, p8 > f)
        True True False True True True
        >>> print(p8 > b, p1 > b)
        True False
        """
        assert self.value != Rank.FLAG and self.value != Rank.BOMB, "Attacker cant be stationary"
        if other.value == Rank.FLAG:
            return True
        if self.value == Rank.SPY:
            return other.value == Rank.MARSHALL
        if other.value == Rank.SPY:
            return True
        if other.value == Rank.BOMB:
            return self.value == Rank.MINER
        # Use less than since strong piece has lower rank value
        return self.value < other.value

    def __ge__(self, other):
        return self.value == other.value or self > other

    def __str__(self) -> str:
        return str(self.value)

    def __hash__(self):
        return hash(str(self.value))

    def get_file_str(self) -> str:
        r""" Identifier for rank entries in the board and state files """
        return "Rank_" + str(self)

    @staticmethod
    def get_from_file(line_header: str) -> 'Rank':
        r""" Convert a line header from a board or state file to a corresponding Rank object. """
        spl = line_header.split("_", maxsplit=1)
        try:
            return Rank(int(spl[1]))
        except ValueError:
            return Rank(spl[1])


class Piece:
    r""" Encapsulates a single piece.  Each individual piece is a separate object """
    _next_id = 0

    def __init__(self, color: Color, rank: Rank, loc: Location):
        r"""
        :param color: New piece's color
        :param rank: New piece's rank
        :param loc: Location of the new piece
        """
        self._color = color
        self._rank = rank
        self._loc = loc
        self._id = Piece._next_id
        Piece._next_id += 1

    @property
    def rank(self) -> Rank: return self._rank  # pylint: disable=missing-docstring, invalid-name

    @property
    def color(self) -> Color: return self._color  # pylint: disable=missing-docstring, invalid-name

    @property
    def loc(self) -> Location:
        r""" Accessor for the location of the piece """
        return self._loc

    @loc.setter
    def loc(self, loc: Location) -> None:
        r""" Mutator for the piece's location """
        self._loc = loc

    def is_immobile(self) -> bool:
        r""" Returns True if piece cannot move """
        return self._rank.is_immobile()

    def is_scout(self) -> bool:
        r""" Returns True if the piece is a scout """
        return self._rank == Rank.scout()

    def __gt__(self, other: 'Piece') -> bool:
        assert self.color != other.color, "Player cannot attack itself"
        return self.rank > other.rank

    def __eq__(self, other: 'Piece') -> bool:
        # Cannot perform assert check below because of equivalence check of Move objects
        # assert self.color != other.color, "Player cannot attack itself"
        return self.rank == other.rank

    def __hash__(self) -> int:
        return hash(self._id)
