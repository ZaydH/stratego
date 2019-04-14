# -*- coding: utf-8 -*-
r"""
    stratego.__init__
    ~~~~~~~~~~~~~~~~~

    Master game file.

    :copyright: (c) 2019 by Steven Walton and Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""
from pathlib import Path
from typing import Union, Tuple

# import matplotlib
from stratego.location import Location
from .move import Move
from .board import Board
from .printer import Printer
from .state import State


class Game:
    r""" Encapsulates an active Stratego game """
    def __init__(self, board_path: Union[Path, str], state_path: Union[Path, str],
                 visibility: Printer.Visibility):
        r"""
        :param board_path: Path to the file specifying the board
        :param state_path: Path to the file specifying the game state
        :param visibility: Specifies whose pieces are visible
        """
        if isinstance(state_path, str): state_path = Path(state_path)

        self._brd = Board.importer(board_path)
        Move.set_board(self._brd)

        self._state = State.importer(state_path, self._brd, visibility)

    def move(self, cur_loc: Tuple[int, int], new_loc: Tuple[int, int]) -> bool:
        r"""
        Move the piece at \p cur_loc to \p new_loc

        :param cur_loc: Location of the current piece
        :param new_loc: New location for the piece at \p cur_loc
        :return: True if the move was successful
        """
        orig, new = Location(cur_loc[0], cur_loc[1]), Location(new_loc[0], new_loc[1])
        p = self._state.next_player.get_piece_at_loc(orig)
        if p is None:
            return False

        other = self._state.get_other_player(self._state.next_player)
        attacked = other.get_piece_at_loc(new)
        m = Move(p, orig, new, attacked)
        if not m.verify() or m.piece.color != self._state.next_color:
            return False

        self._state.update(m)
        return True

    def display_current(self):
        r""" Displays the current state of the game to the console """
        print(self._state.write_board())
