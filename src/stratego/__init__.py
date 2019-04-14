# -*- coding: utf-8 -*-
r"""
    stratego.__init__
    ~~~~~~~~~~~~~~~~~

    Master game file.

    :copyright: (c) 2019 by Steven Walton and Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""
from pathlib import Path
from typing import Union

# import matplotlib
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

        self._state = State.importer(state_path, self._brd)
        self._printer = Printer(self._brd, self._state.red.pieces(),
                                self._state.blue.pieces(), visibility)

    def move(self, m: Move) -> None:
        r"""

        :param m: Move to perform
        """
        assert m.verify()
        assert m.piece.color == self._state.next_player(), "Not player's turn to move"
        # ToDo Ensure update accounts for blocked scout moves
        pass

    def display_current(self):
        r""" Displays the current state of the game to the console """
        print(self._printer.write())
