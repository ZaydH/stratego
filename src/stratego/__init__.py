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
from enum import Enum
from pathlib import Path
from typing import Union

import matplotlib

from .location import Location
from .piece import Color, Rank, Piece
from .board import Board
from .player import Player


def setup_logger(quiet_mode: bool = False, filename: str = "test.log",
                 log_level: int = logging.DEBUG):
    r"""
    Logger Configurator

    Configures the test logger.

    :param quiet_mode: True if quiet mode (i.e., disable logging to stdout) is used
    :param filename: Log file name
    :param log_level: Level to log
    """
    date_format = '%m/%d/%Y %I:%M:%S %p'  # Example Time Format - 12/12/2010 11:46:36 AM
    format_str = '%(asctime)s -- %(levelname)s -- %(message)s'
    logging.basicConfig(filename=filename, level=log_level, format=format_str, datefmt=date_format)

    # Also print to stdout
    if not quiet_mode:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(format_str)
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)

    # Matplotlib clutters the logger so change its log level
    # noinspection PyProtectedMember
    matplotlib._log.setLevel(logging.INFO)  # pylint: disable=protected-access

    logging.info("******************* New Run Beginning *****************")


class Game:
    r""" Encapsulates an active Stratego game """
    def __init__(self, board_path: Union[Path, str], state_path: Union[Path, str]):
        if isinstance(state_path, str): state_path = Path(state_path)
        self._brd = Board.importer(board_path)
        self._state = State.importer(state_path, self._brd)

    def move(self):
        pass


class State:
    r""" Stores the current state of the game """
    SEP = "|"

    class ImporterKeys(Enum):
        r""" Keys used in the State file """
        next_turn = "NextTurn"
        piece = "Piece"

    def __init__(self, brd: Board):
        self._next = None
        self._brd = brd
        self._red = Player(Color.RED)
        self._blue = Player(Color.BLUE)

    def next_player(self):
        r""" Accessor for player whose turn is next """
        return self._next

    @staticmethod
    # pylint: disable=protected-access
    def importer(file_path: Union[Path, str], brd: Board) -> 'State':
        r"""
        Factory method to construct \p State objects from a file.

        :param file_path: File defining the current state
        :param brd: Board corresponding to the state to the current state
        :return: Constructed \p State object
        """
        if isinstance(file_path, str): file_path = Path(file_path)
        try:
            with open(file_path, "r") as f_in:
                lines = f_in.readlines()
        except IOError:
            raise IOError("Unable to read state file \"%s\"" % str(file_path))

        state = State(brd)
        for line in lines:
            line = line.strip()
            if line[0] == "#":
                continue
            spl = line.split(State.SEP)
            # Import next player line
            if spl[0] == State.ImporterKeys.next_turn.value:
                assert len(spl) == 2, "Invalid number entries for next player"
                assert state._next is None, "Duplicate next player line"
                state._next = Color[spl[1].upper()]  # autoconverts string to enum
            elif spl[0] == State.ImporterKeys.piece.value:
                assert len(spl) == 4, "Invalid number of entries for piece line"
                p = Piece(Color[spl[1]], Rank(spl[2]), Location.parse(spl[3]))
                plyr = state._red if p.color == Color.RED else state._blue
                plyr.add_piece(p)
            else:
                raise ValueError("Unparseable file file \"%s\"" % line)
        assert state._is_valid(), "Input state is invalid"
        return state

    def _is_valid(self) -> bool:
        r""" Simple sanity check the state is valid """
        res = True

        # Check no duplicate locations. Intersection of the sets means a duplicate
        if self._red.piece_locations() & self._blue.piece_locations():
            logging.error("Duplicate piece locations")
            res = False

        # Check pieces conform to PieceSet field in Board class
        for plyr in [self._red, self._blue]:
            if not plyr.verify_piece_set(self._brd.piece_set):
                logging.error("Player %s has invalid piece information", plyr.color.name)
                res = False

        return res

    @staticmethod
    def _print_template_file(file_path: Union[Path, str]) -> None:
        r"""
        Constructs a template state file for demonstration purposes and to allow the user to fill
        in manually.

        :param file_path: Path where to write the state file
        """
        lines = [["# Specify player whose turn is next"],
                 [State.ImporterKeys.next_turn.value, "RED or BLUE"]]

        for color in [Color.RED.name, Color.BLUE.name]:
            lines.append([f"# Add {color} piece information"])
            for _ in range(3):
                lines.append([State.ImporterKeys.piece.value, color,
                              "RANK", Location.file_example_str()])

        # Write file to disk
        if not isinstance(file_path, Path): file_path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True)
        with open(file_path, "w+") as f_out:
            f_out.write("# State template file generated by \"State._print_template_file\"\n"
                        "# Lines beginning with \"#\" are comments\n")
            f_out.write("\n".join([State.SEP.join(line) for line in lines]))
