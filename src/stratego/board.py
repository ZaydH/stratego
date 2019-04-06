# -*- coding: utf-8 -*-
r"""
    stratego.board
    ~~~~~~~~~~~~~~

    Define the \p Board class.

    :copyright: (c) 2019 by Steven Walton & Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""
import logging
import re
from typing import Union
from pathlib import Path
from enum import Enum

from piece import Rank


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

    def __eq__(self, other) -> bool:
        return self.r == other.r and self.c == other.c

    def __hash__(self) -> int:
        if self._hash is None: self._hash = hash((self.r, self.c))
        return self._hash

    def down(self) -> 'Location':
        r""" Build location below the implicit one """
        return Location(self.r - 1, self.c)

    def up(self) -> 'Location':  # pylint: disable=invalid-name
        r""" Build location above the implicit one """
        return Location(self.r + 1, self.c)

    def left(self) -> 'Location':
        r""" Build location to the left of the implicit one """
        return Location(self.r, self.c - 1)

    def right(self) -> 'Location':
        r""" Build location to the right of the implicit one """
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


class Board:
    r""" Encapsulates the default board and piece information for the Stratego board """
    SEP = "|"

    class ImporterKeys(Enum):
        r""" Line header keys used in the board file import """
        board_dim = "BoardDim"
        blocked_loc = "BlockedLoc"

    class PieceSet:
        r""" Set of pieces """
        def __init__(self):
            r"""
            >>> ps = Board.PieceSet(); print(ps.tot_count)
            0
            >>> ps.set_rank_count(Rank(Rank.SPY), 4); ps.set_rank_count(Rank(4), 5)
            >>> print(ps.tot_count)
            9
            >>> ps.set_rank_count(Rank(Rank.SPY), 1)
            >>> print(ps.tot_count)
            6
            """
            self._counts = dict()
            self._n = 0

        def set_rank_count(self, rank: Rank, count: Union[str, int]) -> None:
            r""" Mutator for the number of pieces of specified rank. """
            if rank in self._counts:
                self._n -= self._counts[rank]

            if not isinstance(count, int): count = int(count)
            assert count >= 0, "Rank count set below zero"
            self._counts[rank] = count
            self._n += count

        def get_rank_count(self, rank: Rank) -> int:
            r""" Accessor for number of pieces of the specified rank """
            assert rank in self._counts, "Getting rank count but not set"
            return self._counts[rank]

        @property
        def tot_count(self) -> int:
            r""" Number of initial pieces PER PLAYER """
            return self._n

        def has_all_ranks(self) -> bool:
            r""" Verify all ranks are accounted for """
            return len(self._counts) == len(Rank.get_all())

        def exists(self, rank: Rank) -> bool:
            r""" Returns True if rank already has a value """
            return rank in self._counts

    def __init__(self):
        self._rows = self._cols = None
        self._blocked = set()  # locations within board boundaries pieces cannot enter
        self._piece_set = Board.PieceSet()

    @property
    def num_rows(self) -> int:
        r""" Accessor for number of rows on the board """
        return self._rows

    @property
    def num_cols(self) -> int:
        r""" Accessor for number of columns on the board """
        return self._cols

    @staticmethod
    # pylint: disable=protected-access
    def importer(file_path: Union[Path, str]) -> 'Board':
        r"""
        Board generator that follows the Factory Method pattern.

        :param file_path: Path to a board file.
        :return: Board with parameters in in \p file_path
        """
        if isinstance(file_path, str): file_path = Path(file_path)

        try:
            with open(file_path, "r") as f_in:
                lines = f_in.readlines()
        except IOError:
            raise ValueError("Unable to read board file \"%s\"" % str(file_path))

        brd = Board()
        rank_headers = [r.board_file_str() for r in Rank.get_all()]
        for line in lines:
            spl = line.split(Board.SEP)
            # Parse the board dimension information
            if spl[0] == Board.ImporterKeys.board_dim.value:
                assert len(spl) == 3, "Invalid board dimension information"
                assert brd._rows is None and brd._cols is None, "Duplicate dimensions line"
                brd._rows, brd._cols = int(spl[1]), int(spl[2])
            # Parse the board locations where a piece cannot be placed (e.g., lake)
            elif spl[0] == Board.ImporterKeys.blocked_loc.value:
                assert len(spl) > 1, "No blocked location specified"
                for loc_str in spl[1:]:
                    loc = Location.parse(loc_str)
                    assert loc.r < brd.num_rows and loc.c < brd.num_cols, "Outside board boundary"
                    assert loc not in brd._blocked, "Duplicate blocked location"
                    brd._blocked.add(loc)
            # Parse the (original) number of pieces for a given rank
            elif spl[0] in rank_headers:
                rank = Rank.get_from_board_file(spl[0])
                assert not brd._piece_set.exists(rank), "Duplicate rank count in board file"
                brd._piece_set.set_rank_count(rank, spl[1])
            else:
                raise ValueError("Unparseable file line \"%s\"" % line)
        # Sanity check the configuration
        assert brd._is_valid()  # pylint: disable=protected-access
        return brd

    def can_piece_occupy(self, loc: Location) -> bool:
        r"""
        Checks whether a piece could enter the specified location. The location could be illegal
        due to be being off the board or being impassable (e.g., due to a lake).

        :param loc: Location of interest
        :return: True if the location supports piece placement
        """
        if loc.r < 0 or loc.r >= self.num_rows or loc.c < 0 or loc.c >= self.num_cols:
            return False
        return loc not in self._blocked

    def _is_valid(self):
        r""" Returns true if the board has no (obvious) errors """
        res = True
        if self._rows is None or self._cols is None or self.num_rows <= 0 or self.num_cols <= 0:
            logging.warning("Invalid number of rows/column in board.")
            res = False
        # Multiply number of pieces by two since two players
        if self.num_rows * self.num_cols - len(self._blocked) < 2 * self._piece_set.tot_count:
            logging.warning("More pieces on board than open spaces")
            res = False
        # Make sure no piece was excluded from the board file
        if not self._piece_set.has_all_ranks():
            logging.warning("Some ranks missing from the Board's PieceSet")
            res = False
        # Ensure number of pieces aligns with a complete row for each player
        for row, inc in ((0, 1), (self.num_rows - 1, -1)):
            rem_count = self._piece_set.tot_count
            while rem_count > 0 and 0 <= row < self.num_rows:
                for col in range(0, self.num_cols):
                    if Location(row, col) not in self._blocked:
                        rem_count -= 1
                row += inc
            if rem_count != 0:
                logging.warning("Piece count does not align with number of rows")
                res = False
                break
        return res

    @staticmethod
    def _print_template_file(file_path: Union[Path, str]) -> None:
        r"""
        Creates a template file a user could fill-in to create a board

        :param file_path: Location to write the template file
        """
        # noinspection PyListCreation
        lines = [[Board.ImporterKeys.board_dim.value, "NUM_ROWS", "NUM_COLS"]]

        lines.append([Board.ImporterKeys.blocked_loc.value, Location.board_file_example()])
        for _ in range(0, 2):
            lines[-1].append(Location.board_file_example())
        lines[-1].append("...")
        # Show multiple lines is supported
        lines.append([Board.ImporterKeys.blocked_loc.value, Location.board_file_example()])

        for rank in Rank.get_all():
            lines.append([rank.board_file_str(), "STARTING_COUNT"])

        file_path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True)
        with open(file_path, "w+") as f_out:
            f_out.write("\n".join([Board.SEP.join(l) for l in lines]))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
