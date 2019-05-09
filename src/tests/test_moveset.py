# -*- coding: utf-8 -*-
r"""
    tests.test_moveset
    ~~~~~~~~~~~~~~~~~~

    Tests the construction of the \p MoveSet class.

    :copyright: (c) 2019 by Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""

from stratego.state import State

from testing_utils import STATES_PATH, STD_BRD


def test_basic_board():
    r""" Verify the basic components work """
    state = State.importer(STATES_PATH / "moveset_basic.txt", STD_BRD)
    for plyr in [state.red, state.blue]:
        assert len(plyr.move_set) == plyr.num_pieces

    state = State.importer(STATES_PATH / "moveset_basic_immobile.txt", STD_BRD)
    for plyr in [state.red, state.blue]:
        assert plyr.num_pieces == 4
        assert len(plyr.move_set) == 3

    state = State.importer(STATES_PATH / "moveset_basic_attack.txt", STD_BRD)
    for plyr in [state.red, state.blue]:
        assert plyr.num_pieces == 2

    state = State.importer(STATES_PATH / "moveset_basic_blocked.txt", STD_BRD)
    for plyr in [state.red, state.blue]:
        assert plyr.num_pieces == 3
        assert len(plyr.move_set) == 4
        assert len(plyr.move_set) == 4


def test_moveset_scout():
    r""" Verify the moveset with scout """
    state = State.importer(STATES_PATH / "moveset_scout_basic.txt", STD_BRD)
    # Only pieces that can move is the one scout
    assert len(state.red.move_set) == 18
    assert len(state.blue.move_set) == 4

    state = State.importer(STATES_PATH / "moveset_scout_blocked.txt", STD_BRD)
    assert state.red.num_pieces == 4
    assert state.blue.num_pieces == 3
    # Verify the RED piece moves
    num_red_mv = ((STD_BRD.num_rows - 1 - 1) + 1  # Piece in (4, 0) (Subtract one due to (0,0) flag
                  + (STD_BRD.num_rows - 1) + 1  # Piece in (5, 1)
                  + 4)  # Piece in (1, 4)
    assert len(state.red.move_set) == num_red_mv
    # Verify the BLUE piece moves
    num_blue_mv = ((3 + 1) + 1  # Piece at (4, 4)
                   + 4)  # Piece at (6,4)
    assert len(state.blue.move_set) == num_blue_mv
