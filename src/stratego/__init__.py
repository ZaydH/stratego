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
from typing import Union, Tuple, Optional

# import matplotlib
from stratego.agent import Agent
from stratego.location import Location
from stratego.utils import StrOrPath
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
            print("Player: %s (Rank %s) moved from %s to %s\n"
                  % (m.piece.color.name, m.piece.rank, m.orig, m.new))
            sys.stdout.flush()

        if not self._state.update(m): return False

        if display:
            print("\n".join([self._state.write_board(), "\n"]))
            sys.stdout.flush()
        return True

    def two_agent_automated(self, a1: Agent, a2: Agent, wait_time: float = 0,
                            display: bool = False, max_num_moves: Optional[int] = None,
                            moves_output_file: Optional[StrOrPath] = None) -> None:
        r"""
        Simple interface to play

        :param a1: First agent (can be of either color)
        :param a2: Second automated game playing agent (must be opposite color of \p a1)
        :param wait_time: Time (in seconds) to wait between moves.  Must be non-negative
        :param display: If True, display information about the move and the board
        :param max_num_moves: Maximum number of moves to perform before existing.
        :param moves_output_file: Path to write the moves made
        """
        assert a1.color != a2.color, "Both agents cannot be the same color"

        if wait_time < 0:
            raise ValueError("wait_time must be non-negative")
        if max_num_moves is None: max_num_moves = sys.maxsize
        if max_num_moves <= 0:
            raise ValueError("Maximum number of moves must be non-negative")

        if display:
            self.display_current()
            print("")
            if wait_time > 0: time.sleep(wait_time)

        w_move = moves_output_file is not None
        if w_move:
            f_out = open(moves_output_file, "w+")
            f_out.write("PlayerColor,StartLoc,EndLoc")

        num_moves = 0
        cur, other = (a1, a2) if a1.color == self._state.next_player.color else (a2, a1)
        while num_moves < max_num_moves and not self._state.is_game_over():
            m = cur.get_next_move()
            self.play_move(m, display)

            if w_move:
                # noinspection PyUnboundLocalVariable
                f_out.write("\n%s,%s,%s" % (cur.color.name, m.orig, m.new))

            if wait_time > 0: time.sleep(wait_time)
            cur, other = other, cur
            num_moves += 1

        if w_move:
            f_out.close()
        if self._state.is_game_over():
            if display:
                print("Game over.  Player", other.color.name, "won")
                print("Number of Moves:", num_moves)
        elif num_moves == max_num_moves:
            print("Maximum number of moves %d reached. Exiting." % max_num_moves)

    def display_current(self):
        r""" Displays the current state of the game to the console """
        print(self._state.write_board())
