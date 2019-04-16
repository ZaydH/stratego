import pytest

from stratego import Move
from stratego.location import Location
from stratego.move import MoveStack
from stratego.state import State

from testing_utils import STATES_PATH, STD_BRD, substr_in_err


def test_duplicate_loc_in_state():
    r""" Verify that a \p State file with two pieces in same location raises an error """
    for dup_file in ["duplicate_loc_red.txt", "duplicate_loc_diff_color.txt"]:
        duplicate_path = STATES_PATH / dup_file
        assert duplicate_path.exists(), "Duplicate file path does not exist"

        with pytest.raises(Exception):
            State.importer(duplicate_path, STD_BRD)


def test_no_flag():
    r""" Verify an error is raised if the file has no flag """
    # Verify the "clean" passes
    path = STATES_PATH / "no_flag_clean.txt"
    assert path.exists(), "No flag test file does not exist"
    State.importer(path, STD_BRD)

    # Verify no flag checks are done for both players
    for file in ["no_flag_red.txt", "no_flag_blue.txt"]:
        path = STATES_PATH / file
        assert path.exists(), "No flag test file does not exist"

        with pytest.raises(Exception):
            State.importer(path, STD_BRD)


# noinspection PyProtectedMember
def test_state_moves():
    path = STATES_PATH / "state_move_verify.txt"
    assert path.exists(), "Move verify file does not exist"

    state = State.importer(path, STD_BRD)

    # Standardize assert tests
    def _test_num_pieces_and_move_set_size(_num_red_p, _num_blue_p, _num_red_mv, _num_blue_mv):
        assert state.red.num_pieces == _num_red_p
        assert state.blue.num_pieces == _num_blue_p
        assert len(state.red.move_set) == _num_red_mv
        assert len(state.blue.move_set) == _num_blue_mv

    # Verify initial state matches expectations
    _test_num_pieces_and_move_set_size(7, 7, 4 + 3, 4 + 3)

    move_stack = MoveStack()
    # Define a series of moves.  Entries in each tuple are:
    #   0: Original piece location
    #   1: Piece new location
    #   2: Number of red pieces
    #   3: Number of blue pieces
    #   4: Size of the red move set
    #   %: Size of the blue move set
    move_list = [((0, 1), (1, 1), 7, 7, 12, 7),
                 ((9, 1), (8, 1), 7, 7, 12, 12),
                 ((1, 1), (2, 1), 7, 7, 12, 12),
                 ((8, 1), (7, 1), 7, 7, 12, 12),
                 ((2, 1), (3, 1), 7, 7, 12, 12),
                 ((7, 1), (6, 1), 7, 7, 12, 12),
                 ((3, 1), (4, 1), 7, 7, 11, 12),  # One less due to blocked by (4, 2)
                 ((6, 1), (5, 1), 7, 7, 11, 11),  # One less due to blocked by (5, 2)
                 ((4, 1), (5, 1), 6, 6, 8, 8),  # Both lost piece in battle
                 ((9, 3), (6, 3), 6, 6, 8, 18),  # Move blue scout
                 ((0, 3), (3, 3), 6, 6, 18, 18),  # Move red scout
                 ((6, 3), (6, 5), 6, 6, 18, 23),  # Move blue scout
                 ((3, 3), (3, 5), 6, 6, 20, 20),  # Move red scout
                 ((6, 5), (6, 4), 6, 6, 23, 23),  # Move blue scout
                 ((3, 5), (9, 5), 6, 5, 16, 22),  # Red scout attack blue spy
                 ((6, 4), (0, 4), 6, 4, 16, 5)  # Blue scout attack red bomb
                 ]
    printer_out = []
    for orig, new, num_red_p, num_blue_p, num_red_mv, num_blue_mv in move_list:
        orig, new = Location(orig[0], orig[1]), Location(new[0], new[1])
        p = state.next_player.get_piece_at_loc(orig)
        assert p is not None
        attacked = state.get_other_player(state.next_player).get_piece_at_loc(new)

        move_stack.push(Move(p, orig, new, attacked))
        assert state.update(move_stack.top())
        assert state._printer._is_loc_empty(orig)

        _test_num_pieces_and_move_set_size(num_red_p, num_blue_p, num_red_mv, num_blue_mv)
        printer_out.append(state.write_board())

    # Try to move red bomb then the red flag
    for orig in [Location(0, 4), Location(0, 6)]:
        p = state.next_player.get_piece_at_loc(orig)
        assert p is not None
        for new in [orig.left(), orig.right]:
            attacked = state.get_other_player(state.next_player).get_piece_at_loc(new)

            with pytest.raises(Exception):
                Move(p, orig, new, attacked)

    # Verify Undo
    for i in range(2, len(move_list) + 1):
        _, _, num_red_p, num_blue_p, num_red_mv, num_blue_mv = move_list[-i]
        state.undo()

        assert state.write_board() == printer_out[-i], "Printer mismatch after do/undo"
        _test_num_pieces_and_move_set_size(num_red_p, num_blue_p, num_red_mv, num_blue_mv)

