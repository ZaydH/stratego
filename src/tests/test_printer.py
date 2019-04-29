from stratego.location import Location
from stratego.piece import Color, Piece, Rank
from stratego.printer import Printer

from testing_utils import build_test_board


# pylint: disable=protected-access
# noinspection PyProtectedMember
def test_printer_piece_movement():
    r""" Verify piece movement as part of the \p Printer """
    brd = build_test_board(5, 5)
    p = Printer(brd, {Piece(Color.RED, Rank(1), Location(2, 1))},
                {Piece(Color.BLUE, Rank(2), Location(3, 2))}, Printer.Visibility.RED)
    assert p._is_loc_empty(Location(0, 0))
    assert not p._is_loc_empty(Location(3, 2))

    p.move_piece(Location(3, 2), Location(2, 4))
    assert not p._is_loc_empty(Location(2, 4))
    assert p._is_loc_empty(Location(3, 2))

    p.delete_piece(Location(2, 4))
    assert p._is_loc_empty(Location(2, 4))


# noinspection PyProtectedMember
def test_printer_visibility():
    r""" Verify the visibility settings of the \p Printer class """
    brd = build_test_board(4, 4)
    p = Printer(brd, set(), set(), Printer.Visibility.NONE)
    assert not p._is_visible(Color.RED)
    assert not p._is_visible(Color.BLUE)

    p = Printer(brd, set(), set(), Printer.Visibility.RED)
    assert p._is_visible(Color.RED)
    assert not p._is_visible(Color.BLUE)

    p = Printer(brd, set(), set(), Printer.Visibility.ALL)
    assert p._is_visible(Color.RED)
    assert p._is_visible(Color.BLUE)
