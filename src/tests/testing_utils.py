from typing import Optional, Set

from stratego.board import Board
from stratego.location import Location


def build_test_board(num_rows: int, num_cols: int,
                     blocked: Optional[Set[Location]] = None) -> Board:
    r""" Create a skeleton \p Board object for TESTING PURPOSES ONLY """
    brd = Board()
    brd._rows, brd._cols = num_rows, num_cols
    brd._blocked = blocked if blocked is not None else {}
    return brd
