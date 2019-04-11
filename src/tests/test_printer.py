from stratego.location import Location
from stratego.piece import Color, Piece, Rank
from stratego.printer import Printer


# pylint: disable=protected-access
def test_piece_movement():
    r""" Verify piece movement as part of the \p Printer """
    p = Printer(5, 5, {}, {Piece(Color.RED, Rank(1), Location(2, 1))},
                {Piece(Color.BLUE, Rank(2), Location(3, 2))}, Printer.Visibility.RED)
    assert p._is_loc_empty(Location(0, 0))
    assert not p._is_loc_empty(Location(3, 2))

    p.move_piece(Location(3, 2), Location(2, 4))
    assert not p._is_loc_empty(Location(2, 4))
    assert p._is_loc_empty(Location(3, 2))

    p.delete_piece(Location(2, 4))
    assert p._is_loc_empty(Location(2, 4))
