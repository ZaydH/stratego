from pathlib import Path
from typing import Optional, Set

from stratego.board import Board
from stratego.location import Location


BOARDS_PATH = Path("tests/boards")
STATES_PATH = Path("tests/states")

# Standard board file
STD_BRD = Board.importer(BOARDS_PATH / "standard.txt")


def build_test_board(num_rows: int, num_cols: int,
                     blocked: Optional[Set[Location]] = None) -> Board:
    r""" Create a skeleton \p Board object for TESTING PURPOSES ONLY """
    brd = Board()
    brd._rows, brd._cols = num_rows, num_cols
    brd._blocked = blocked if blocked is not None else set()
    return brd


# noinspection PyUnresolvedReferences
def substr_in_err(substr: str, e_info: 'ExceptionInfo') -> bool:
    r"""
    Case insensitive check of cubstring in the error raised

    :param substr: String to check if contained in error message
    :param e_info: Error information
    :return: True if the substr appears in \p e_info's error
    """
    return substr.lower() in str(e_info.value).lower()
