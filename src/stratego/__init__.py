# -*- coding: utf-8 -*-
r"""
    stratego.__init__
    ~~~~~~~~~~~~~~~~~

    Implements the master \p Game class.

    :copyright: (c) 2019 by Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""
import logging
import re
import sys
import time
from pathlib import Path
from typing import Union, Tuple, Optional

# import matplotlib
from .agent import Agent
from .board import Board
from .location import Location
from .move import Move
from .player import Player
from .printer import Printer
from .state import State
from .utils import PathOrStr


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

    @property
    def red(self) -> Player:
        r""" Accessor for the RED player """
        return self._state.red

    @property
    def blue(self) -> Player:
        r""" Accessor for the BLUE player """
        return self._state.blue

    @property
    def board(self) -> Board:
        r""" Accessor to the board used for the game """
        return self._brd

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
            raise ValueError("No %s piece at location (%d,%d)", *fields)

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
                            moves_output_file: Optional[PathOrStr] = None) -> None:
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
            # Add header row for improved file readability
            f_out.write("# PlayerColor,StartLoc,EndLoc")

        num_moves = 0
        cur, other = (a1, a2) if a1.color == self._state.next_player.color else (a2, a1)
        while num_moves < max_num_moves and not self._state.is_game_over():
            m = cur.get_next_move()
            if w_move:
                # noinspection PyUnboundLocalVariable
                f_out.write("\n%s,%s,%s" % (cur.color.name, m.orig, m.new))
            self.play_move(m, display)

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

    def _execute_move_file(self, moves_file: PathOrStr, display_after_move: bool = False) -> None:
        r"""
        Debug method to execute a set of moves in file \p moves_file.  Each move is specified on a
        separate line with the format:

            \p PIECE_COLOR,(ORIG_ROW,ORIG_COL),(NEW_ROW,NEW_COL)

        Blank lines or those starting with \"#\" are ignored.

        :param moves_file: File containing the moves to be performed
        :param display_after_move: If True, print the board state after each move
        """
        try:
            with open(moves_file, "r") as f_in:
                lines = f_in.read().splitlines()
        except IOError:
            raise IOError("Unable to read moves file: \"%s\"" % str(moves_file))
        # Match a color string then the move ",(row,col)" twice
        move_regex = re.compile(r"(\w+),\((\d+),(\d+)\),\((\d+),(\d+)\)")
        for line in lines[1:]:  # Skip header line
            line = line.strip()
            # Skip blank lines or those starting with "#"
            if not line or line[0] == "#": continue
            if not move_regex.match(line): raise ValueError("Unable to parse move \"%s\"" % line)
            res = re.findall(r"\d+", line)
            locs = [(int(res[i]), int(res[i+1])) for i in range(0, 3, 2)]
            self.move(*locs)
            if display_after_move:
                self.display_current()

    def display_current(self):
        r""" Displays the current state of the game to the console """
        print(self._state.write_board())
