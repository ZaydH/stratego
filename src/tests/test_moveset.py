from stratego.board import Board
from stratego.location import Location
from stratego.player import MoveSet
from stratego.state import State

from testing_utils import BOARDS_PATH, STATES_PATH


# Create a dummy board for use in comparison in the movements
num_brd_rows = num_brd_cols = 10
blocked_loc = Location(3, 2)


def test_basic_board():
    brd = Board.importer(BOARDS_PATH / "standard.txt")
    MoveSet.set_board(brd)

    state = State.importer(STATES_PATH / "moveset_state.txt", brd)

    for plyr in [state.red, state.blue]:
        num_pieces = sum(1 for _ in plyr.pieces())
        assert len(plyr.move_set) == num_pieces
