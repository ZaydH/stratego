# -*- coding: utf-8 -*-
r"""
    stratego.__init__
    ~~~~~~~~~~~~~~~~~

    Master game file.

    :copyright: (c) 2019 by Steven Walton and Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""
import logging
import sys
import time
from pathlib import Path
from typing import Union, Tuple

# import matplotlib
from stratego.agent import Agent
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
            fields = (self._state.next_color.name, orig.r, orig.c)
            logging.warning("No %s piece at location (%d,%d)", *fields)
            return False

        other = self._state.get_other_player(self._state.next_player)
        attacked = other.get_piece_at_loc(new)
        m = Move(p, orig, new, attacked)
        return self.play_move(m)

    def play_move(self, m: Move, display: bool = False) -> bool:
        r"""
        Performs the specified \p Move
        :param m: Move to be performed
        :param display: If True, display information about the move and the board
        :return: True if the move was successful
        """
        try: m.verify()
        except KeyError as e_info:
            logging.error(str(e_info))
            return False

        if m.piece.color != self._state.next_color:
            logging.error("Moved piece's color did not match the expected color")
            return False

        if display:
            print("Player: %s moved from %s to %s\n" % (m.piece.color, m.orig, m.new))
            sys.stdout.flush()

        if not self._state.update(m): return False

        if display:
            print("\n".join([self._state.write_board(), "\n"]))
            sys.stdout.flush()
        return True

    def two_agent_automated(self, a1: Agent, a2: Agent, wait_time: float = 0,
                            display: bool = False) -> None:
        r"""
        Simple interface to play

        :param a1: First agent (can be of either color)
        :param a2: Second automated game playing agent (must be opposite color of \p a1)
        :param wait_time: Time (in seconds) to wait between moves.  Must be non-negative
        :param display: If True, display information about the move and the board
        :return:
        """
        if wait_time < 0:
            raise ValueError("wait_time must be non-negative")
        assert a1.color != a2.color, "Both agents cannot be the same color"

        if display:
            self.display_current()
            print("")
            if wait_time > 0: time.sleep(wait_time)

        num_moves = 0
        cur, other = (a1, a2) if a1.color == self._state.next_player.color else (a2, a1)
        while not self._state.is_game_over():
            m = cur.get_next_move()
            self.play_move(m, display)
            if wait_time > 0: time.sleep(wait_time)
            cur, other = other, cur
            num_moves += 1

        if display:
            print("Game over.  Player %d won" % other.color.name)
            print("Number of Moves: %d", num_moves)

    def display_current(self):
        r""" Displays the current state of the game to the console """
        print(self._state.write_board())
