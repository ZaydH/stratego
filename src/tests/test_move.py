# -*- coding: utf-8 -*-
r"""
    tests.test_move
    ~~~~~~~~~~~~~~~

    Verifies basic \p Move object construction including verify error conditions actually raise
    errors.

    :copyright: (c) 2019 by Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""

import pytest

from stratego.location import Location
from stratego.move import Move
from stratego.piece import Color, Piece, Rank

from testing_utils import STD_BRD, build_test_board, substr_in_err

Move.set_board(STD_BRD)


def test_verify_neighbor_movements():
    r""" Verifies basic movement of a piece that ensures basic movement works """
    l_orig = Location(1, 1)
    p = Piece(Color.RED, Rank.marshall(), l_orig)

    # Verify all the neighbor locations are valid
    for l_new in l_orig.neighbors():
        Move(p, l_orig, l_new)

    # Verify with attacking
    other = Piece(Color.BLUE, Rank.bomb(), Location(0, 0))
    for l_new in l_orig.neighbors():
        other.loc = l_new
        Move(p, l_orig, l_new, other)


def test_outside_board_moves():
    r""" Verify that the program raises errors when moves are illegal given the board"""
    l_orig = Location(0, 0)

    # Test top and left boundaries
    p = Piece(Color.RED, Rank.marshall(), l_orig)
    for l_new in [l_orig.up(), l_orig.left()]:
        with pytest.raises(Exception) as e_info:
            Move(p, l_orig, l_new)
        assert e_info.type == ValueError

    # Test right and bottom boundaries
    l_orig = Location(STD_BRD.num_rows - 1, STD_BRD.num_cols - 1)
    p = Piece(Color.RED, Rank.marshall(), l_orig)
    for l_new in [l_orig.right(), l_orig.down()]:
        with pytest.raises(Exception) as e_info:
            Move(p, l_orig, l_new)
        assert e_info.type == ValueError

    blocked_loc = Location(4, 2)
    # Verify a blocked location cannot be used as original location
    p = Piece(Color.RED, Rank.marshall(), blocked_loc)
    with pytest.raises(Exception) as e_info:
        Move(p, blocked_loc, blocked_loc.up())
    assert e_info.type == ValueError

    # Verify a blocked location cannot be used as new location
    p = Piece(Color.RED, Rank.marshall(), blocked_loc.up())
    with pytest.raises(Exception) as e_info:
        Move(p, blocked_loc.up(), blocked_loc)
    assert e_info.type == ValueError


def test_attack_color():
    r""" Verify that attacks of different colors either do or do not raise errors """
    # Verify when blue attacks red and red attacks blue, everything works
    loc = Location(0, 0)
    red = Piece(Color.RED, Rank.marshall(), loc)
    blue = Piece(Color.BLUE, Rank.marshall(), loc.down())
    # Red attack blue
    Move(red, loc, blue.loc, blue)
    # Blue attack red
    Move(blue, blue.loc, red.loc, red)

    # Red attack red (exception)
    red2 = Piece(red.color, Rank.flag(), blue.loc)
    with pytest.raises(Exception):
        Move(red, red.loc, red2.loc, red2)
    # Blue attack blue (exception)
    blue2 = Piece(blue.color, Rank.flag(), red.loc)
    with pytest.raises(Exception):
        Move(blue, blue.loc, blue2.loc, blue2)


def test_diagonal_movement():
    r""" Verify diagonal movements fail """
    loc = Location(1, 1)
    move_list = [-1, 0, 1]

    p = Piece(Color.RED, Rank.spy(), loc)
    for row_move in move_list:
        for col_move in move_list:
            man_dist = abs(col_move) + abs(row_move)
            l_new = loc.relative(row_move, col_move)
            # Adjacent moves of (exactly) one are sanity checks and should pass
            if man_dist == 0:
                with pytest.raises(Exception):
                    Move(p, p.loc, l_new)
            # Valid adjacent moves for sanity check
            elif man_dist == 1:
                Move(p, loc, l_new)
            # Diagonal moves should fail
            elif man_dist == 2:
                with pytest.raises(Exception) as e_info:
                    Move(p, p.loc, l_new)
                assert substr_in_err("diagonal", e_info)


def test_scout_movements():
    r""" Verify that only scouts can move multiple squares at once """
    loc = Location(3, STD_BRD.num_cols // 2)

    # Just make sure color of the Scout has not effect (it may be a dumb test)
    for color in [Color.RED, Color.BLUE]:
        scout, miner = Piece(color, Rank.scout(), loc), Piece(color, Rank.miner(), loc)
        for p in [scout, miner]:
            for move_dist in [1, 2]:
                for i in range(2):
                    if i == 0: new_loc = loc.relative(row_diff=move_dist)
                    else: new_loc = loc.relative(col_diff=move_dist)

                    if move_dist == 1 or p.rank == Rank.scout():
                        Move(p, loc, new_loc)
                    else:
                        with pytest.raises(Exception) as e_info:
                            Move(p, p.loc, new_loc)
                        assert substr_in_err("multiple", e_info)


def test_immovable_pieces():
    r""" Verify that an immovable piece raises an error """
    moveable = set(range(Rank.MIN(), Rank.MAX() + 1)) | {Rank.SPY}
    immovable = {Rank.BOMB, Rank.FLAG}
    combined = moveable | immovable

    # Sanity check rank sets
    assert len(moveable) > 0 and len(immovable) > 0, "Both rank sets should be non-empty"
    assert moveable & immovable == set(), "Rank sets not disjoint"
    assert len(combined) == len(Rank.get_all()), "One or more ranks missing"

    loc = Location(0, 0)
    # As a sanity make sure color is irrelvant
    for color in [Color.RED, Color.BLUE]:
        # Verify each rank is correctly marked movable or immovable
        for r in combined:
            p = Piece(color, Rank(r), loc)
            new_loc = loc.down()
            if r in moveable:
                Move(p, loc, new_loc)
            if r in immovable:
                with pytest.raises(Exception):
                    Move(p, p.loc, new_loc)
