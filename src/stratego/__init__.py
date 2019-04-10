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
from pathlib import Path
from typing import Union

# import matplotlib
from .board import Board
from .printer import Printer
from .state import State


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
    # matplotlib._log.setLevel(logging.INFO)  # pylint: disable=protected-access

    logging.info("******************* New Run Beginning *****************")


class Game:
    r""" Encapsulates an active Stratego game """
    def __init__(self, board_path: Union[Path, str], state_path: Union[Path, str],
                 visibility: Printer.Visibility):
        if isinstance(state_path, str): state_path = Path(state_path)

        self._brd = Board.importer(board_path)
        self._state = State.importer(state_path, self._brd)
        self._printer = Printer(self._brd, self._state, visibility)

    def move(self):
        pass

    def display_current(self):
        r""" Displays the current state of the game to the console """
        print(self._printer.write())
