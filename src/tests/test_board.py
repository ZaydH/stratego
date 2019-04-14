from stratego.board import Board
from stratego.location import Location

from testing_utils import build_test_board

# Create a dummy board for use in comparison in the movements
num_brd_rows = num_brd_cols = 10
blocked_loc = Location(5, 5)

brd = build_test_board(num_brd_rows, num_brd_cols, {blocked_loc})  # type: Board


def test_build_test_board():
    r""" Test the function \p build_test_board """
    assert brd.num_rows == num_brd_rows
    assert brd.num_cols == num_brd_cols
    assert {blocked_loc} == brd.blocked


def test_edge_lists_board_corner():
    r""" Verify basic information for two edge lists """
    corners = [Location(0, 0), Location(0, num_brd_cols - 1),
               Location(num_brd_rows - 1, 0), Location(num_brd_rows - 1, num_brd_cols - 1)]
    for c in corners:
        el = brd.to_edge_lists(c)
        num_empty = sum([1 for x in el if len(x) == 0])
        assert num_empty == 2, "Corner to edge list does not have two empty lists"

        # Verify length of the lists
        num_squares = sum([len(x) for x in el])
        expected_count = num_brd_rows + num_brd_cols - 2  # Subtract two to loc itself in each dir
        assert num_squares == expected_count

        num_in_row = len(el.right) + len(el.left)
        assert num_in_row == num_brd_cols - 1
        num_in_col = len(el.up) + len(el.down)
        assert num_in_col == num_brd_rows - 1


def test_edge_lists_blocked_square():
    r""" Verify that the blocked squares are properly handled in \p ToEdgeList construction """
    # To Edge Lists should be blocked
    el = brd.to_edge_lists(blocked_loc.up())
    assert not el.down
    el = brd.to_edge_lists(blocked_loc.right())
    assert not el.left
    el = brd.to_edge_lists(blocked_loc.down())
    assert not el.up
    el = brd.to_edge_lists(blocked_loc.left())
    assert not el.right
