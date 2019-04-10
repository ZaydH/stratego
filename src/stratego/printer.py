from enum import Enum

from sty import fg, bg, ef, rs, RgbFg
from typing import Union

from .board import Board
from .location import Location
from .piece import Color, Piece
from .state import State


class Printer:
    r"""
    Dedicated printer class that caches information about the board so it does not need to be
    regenerated each time.
    """

    SEP = "|"
    EMPTY_LOCATION = " "  # Verify
    HIDDEN = " "
    # HIDDEN = chr(9608)  # White box

    class Visibility(Enum):
        NONE = set()
        RED = {Color.RED}
        BLUE = {Color.BLUE}
        ALL = RED.value | BLUE.value

    def __init__(self, brd: Board, state: State, visibility: 'Printer.Visibility'):
        r"""

        :param brd: Board of interest.  (Assumed immutable.)
        :param state: State of the players.
        :param visibility: Player(s) if any that are visible.
        """
        self._brd = brd
        # Empty board with all pieces blank
        self._piece_state = [[Printer.EMPTY_LOCATION for _ in range(self._brd.num_cols)]
                             for _ in range(self._brd.num_rows)]
        self._visible = visibility.value

    def _is_loc_empty(self, loc: Location) -> bool:
        r""" Returns true if the specified location is empty """
        return self._piece_state[loc.r][loc.c] == Printer.EMPTY_LOCATION

    def delete_piece(self, loc: Location):
        r""" Deletes piece at the specified location """
        assert not self._is_loc_empty(loc), "Tried to delete piece that does not exist"
        self._piece_state[loc.r][loc.c] = Printer.EMPTY_LOCATION

    def move_piece(self, orig: Location, new: Location):
        r"""
        Moves piece from \p orig to \p new.  If \p new has a piece already, that piece is removed
        in the process of the move.
        """
        assert not self._is_loc_empty(orig), "Tried to delete piece that does not exist"
        self._piece_state[new.r][new.c] = self._piece_state[orig.r][orig.c]
        self.delete_piece(orig)

    def _is_visible(self, color: Color) -> bool:
        r""" Returns True if the piece color is visible """
        return color in self._visible

    def _format_piece(self, piece: Piece):
        return "".join([rs.all,
                        bg.red if piece.color == Color.RED else bg.blue,
                        ef.bold, fg.white,  # White writing over the background
                        str(piece) if self._is_visible(piece.color) else Printer.HIDDEN,
                        rs.all  # Go back to normal printing
        ])


class _Console:
    r"""
    Colors class:reset all colors with colors.reset;

    two  sub classes fg for foreground
    and bg for background; use as colors.subclass.colorname.
    i.e. colors.fg.red or colors.bg.green

    also, the generic bold, disable, underline, reverse, strike through,
    and invisible work with the main class i.e. colors.bold

    Source: https://www.geeksforgeeks.org/print-colors-python-terminal/
    """

    class Color:
        black = 0
        red = 1
        green = 2
        orange = 3
        blue = 4
        purple = 5
        cyan = 6
        lightgrey = 7

    @staticmethod
    def _print(cmd: Union[int, str]) -> str:
        r""" Standardizes the """
        return "".join(["\033[", str(cmd), "m"])

    reset = _print("0")
    bold = _print("01")
    disable = _print("02")
    underline = _print("04")
    reverse = _print("07")
    invisible = _print("08")
    strikethrough = _print("09")

    @staticmethod
    def fg(color: _Console.Color):

    class BG:
        @staticmethod
        def _print(cmd: Union[int, str]) -> str:
            r""" Standardizes the """
            return "".join(["\033[", str(cmd), "m"])

        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'
