# -*- coding: utf-8 -*-
r"""
    stratego.board
    ~~~~~~~~~~~~~~

    Implements the \p Board class.

    :copyright: (c) 2019 by Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""
import logging
from typing import List, Set, Union, Tuple
from pathlib import Path
from enum import Enum

from .piece import Rank
from .location import Location


class ToEdgeLists:
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    r""" Contains the two edge lists for a location"""
    def __init__(self, loc: Location):
        r"""
        :param loc: Stores the location the "to edge list" applies
        """
        self.loc = loc
        self.up = None
        self.right = None
        self.down = None
        self.left = None

    def __iter__(self):
        for attr in [self.up, self.right, self.down, self.left]:
            yield attr

    @staticmethod
    def order():
        return [ToEdgeLists.UP, ToEdgeLists.RIGHT, ToEdgeLists.DOWN, ToEdgeLists.LEFT]

    def set(self, dir: int, to_edge_list: List[Location]) -> None:
        if dir == ToEdgeLists.UP: self.up = to_edge_list
        elif dir == ToEdgeLists.RIGHT: self.right = to_edge_list
        elif dir == ToEdgeLists.DOWN: self.down = to_edge_list
        elif dir == ToEdgeLists.LEFT: self.left = to_edge_list
        else:
            raise ValueError("Unknown direction to set")


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
            >>> ps.set_rank_count(Rank.spy(), 4); ps.set_rank_count(Rank(4), 5)
            >>> print(ps.tot_count)
            9
            >>> ps.set_rank_count(Rank.spy(), 1)
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

        def get_rank_count(self, rank: Union[int, str, Rank]) -> int:
            r"""
            Accessor for number of pieces of the specified rank

            :param rank: Piece rank for which to get the count.  If \p rank is of type \p int or
                         \p str, the \p Rank constructor will be called on that object and the
                         result used.
            """
            if isinstance(rank, (int, str)):
                rank = Rank(rank)

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
        self.piece_set = Board.PieceSet()
        self.num_loc = None

    @property
    def num_rows(self) -> int:
        r""" Accessor for number of rows on the board """
        return self._rows

    @property
    def num_cols(self) -> int:
        r""" Accessor for number of columns on the board """
        return self._cols

    @property
    def blocked(self) -> Set[Location]:
        r""" Set of spaces that no piece can enter """
        return self._blocked

    def dim(self) -> Tuple[int, int]:
        r""" Board dimension tuple in the form (#rows, #cols) """
        return self.num_rows, self.num_cols

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
            raise IOError("Unable to read board file \"%s\"" % str(file_path))

        brd = Board()
        rank_headers = [r.get_file_str() for r in Rank.get_all()]
        for line in lines:
            line = line.strip()  # Ignore preceding and trailing whitespace
            # Ignore comment lines
            if line[0] == "#" or not line:
                continue

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
                rank = Rank.get_from_file(spl[0])
                assert not brd.piece_set.exists(rank), "Duplicate rank count in board file"
                brd.piece_set.set_rank_count(rank, spl[1])
            else:
                raise ValueError("Unparseable file line \"%s\"" % line)
        # Sanity check the configuration
        assert brd._is_valid()  # pylint: disable=protected-access
        brd.num_loc = brd.num_rows * brd.num_cols
        return brd

    # noinspection PyProtectedMember
    def can_piece_occupy(self, loc: Location) -> bool:
        r"""
        Checks whether a piece could enter the specified location. The location could be illegal
        due to be being off the board or being impassable (e.g., due to a lake).

        :param loc: Location of interest
        :return: True if the location supports piece placement

        >>> brd = Board(); brd._rows = brd._cols = 4
        >>> print(brd.can_piece_occupy(Location(0,0)), brd.can_piece_occupy(Location(-1,0)))
        True False
        >>> print(brd.can_piece_occupy(Location(0,-1)), brd.can_piece_occupy(Location(4,0)))
        False False
        >>> print(brd.can_piece_occupy(Location(0,4)), brd.can_piece_occupy(Location(1,1)))
        False True
        >>> brd._blocked.add(Location(1,1))
        >>> print(brd.can_piece_occupy(Location(1,1)))
        False
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
        if self.num_rows * self.num_cols - len(self._blocked) < 2 * self.piece_set.tot_count:
            logging.warning("More pieces on board than open spaces")
            res = False
        # Make sure no piece was excluded from the board file
        if not self.piece_set.has_all_ranks():
            logging.warning("Some ranks missing from the Board's PieceSet")
            res = False
        if self.piece_set.get_rank_count(Rank.FLAG) != 1:
            logging.error("Each player can only have exactly one flag")
            res = False
        # # Ensure number of pieces aligns with a complete row for each player
        # for row, inc in ((0, 1), (self.num_rows - 1, -1)):
        #     rem_count = self.piece_set.tot_count
        #     while rem_count > 0 and 0 <= row < self.num_rows:
        #         for col in range(0, self.num_cols):
        #             if Location(row, col) not in self._blocked:
        #                 rem_count -= 1
        #         row += inc
        #     if rem_count != 0:
        #         logging.warning("Piece count does not align with number of rows")
        #         res = False
        #         break
        return res

    def is_inside(self, l: Location) -> bool:
        r"""
        Checks whether the specified location is inside the board

        :param l: Location to check
        :return: True if \p is inside the board dimensions
        """
        return 0 <= l.r < self.num_rows and 0 <= l.c < self.num_cols and l not in self.blocked

    def to_edge_lists(self, loc: Location) -> ToEdgeLists:
        r"""
        Find the locations relative to \p loc that go to the first object up, right, down, and left.
        This is intended for use by scout pieces.

        :param loc: \p Location of interest
        :return: To edge lists for each direction
        """
        e_lists = ToEdgeLists(loc)
        # Order is up, right, down, and left respectively
        mv_list = [(ToEdgeLists.UP, (-1, 0)), (ToEdgeLists.RIGHT, (0, 1)),
                   (ToEdgeLists.DOWN, (1, 0)), (ToEdgeLists.LEFT, (0, -1))]
        for mv_dir, (row_diff, col_diff) in mv_list:
            lst = [loc]
            while True:
                new = lst[-1].relative(row_diff, col_diff)
                # Stop at first block
                if not self.is_inside(new): break
                lst.append(new)
            e_lists.set(mv_dir, lst[1:])
        return e_lists

    @staticmethod
    def _print_template_file(file_path: Union[Path, str]) -> None:
        r"""
        Creates a template file a user could fill-in to create a board

        :param file_path: Location to write the template file
        """
        # noinspection PyListCreation
        lines = [[Board.ImporterKeys.board_dim.value, "NUM_ROWS", "NUM_COLS"]]

        lines.append([Board.ImporterKeys.blocked_loc.value, Location.file_example_str()])
        for _ in range(0, 2):
            lines[-1].append(Location.file_example_str())
        lines[-1].append("...")
        # Show multiple lines is supported
        lines.append([Board.ImporterKeys.blocked_loc.value, Location.file_example_str()])

        for rank in Rank.get_all():
            lines.append([rank.get_file_str(), "STARTING_COUNT"])

        file_path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True)
        with open(file_path, "w+") as f_out:
            f_out.write("# Board file autogenerated with \"Board._print_template_file\"\n"
                        "# Line comments supported by preceding \"#\"\n")
            f_out.write("\n".join([Board.SEP.join(l) for l in lines]))
