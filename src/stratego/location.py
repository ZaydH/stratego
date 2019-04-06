# -*- coding: utf-8 -*-
r"""
    stratego.location
    ~~~~~~~~~~~~~~~~~

    Standardizes the interface for a \p Location object

    :copyright: (c) 2019 by Steven Walton & Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""
import re


class Location:
    r""" Encapsulates a board location """
    def __init__(self, row: int, col: int):
        self._r = row
        self._c = col
        self._hash = None

    @property
    def r(self) -> int:
        r""" Get row of the location """
        return self._r

    @property
    def c(self) -> int:
        r""" Get column of the location """
        return self._c

    def __eq__(self, other: 'Location') -> bool:
        return self.r == other.r and self.c == other.c

    def __hash__(self) -> int:
        if self._hash is None: self._hash = hash((self.r, self.c))
        return self._hash

    def down(self) -> 'Location':
        r""" Build location below the implicit one """
        return Location(self.r + 1, self.c)

    def up(self) -> 'Location':  # pylint: disable=invalid-name
        r""" Build location above the implicit one """
        return Location(self.r - 1, self.c)

    def left(self) -> 'Location':
        r""" Build location to the left of the implicit one """
        return Location(self.r, self.c - 1)

    def right(self) -> 'Location':
        r"""
        Build location to the right of the implicit one

        >>> loc = Location(1,1)
        >>> print(loc.right() == Location(1,2), loc.left() == Location(1, 0))
        True True
        >>> print(loc.down() == Location(2,1), loc.up() == Location(0, 1))
        True True
        """
        return Location(self.r, self.c + 1)

    @staticmethod
    def parse(loc_str: str) -> 'Location':
        r"""
        Converts a string representation of the location to a \p Location.

        :param loc_str: Location in format "(ROW,COL)"
        :return: Corresponding location

        >>> loc = Location(5,4)
        >>> print(loc == Location.parse("(5,4)"), loc == Location.parse("(5,  4)"))
        True True
        >>> print(loc == Location.parse("(4,4)"))
        False
        """
        assert re.match(r"\(\d+,\s*\d+\)", loc_str), "Invalid location string"
        return Location(*[int(x) for x in re.findall(r"\d+", loc_str)])

    @staticmethod
    def board_file_example() -> str:
        r""" Example displaying how the board file blocked location should appear. """
        return "(ROW,COLUMN)"


if __name__ == "__main__":
    import doctest
    doctest.testmod()
