import pytest
from typing import Tuple

from stratego import Move, Game, Printer
from stratego.location import Location
from stratego.move import MoveStack
from stratego.piece import Color
from stratego.player import Player
from stratego.state import State

from testing_utils import STATES_PATH, SMALL_BRD, STD_BRD, substr_in_err


def _get_move_from_player(plyr: Player, _orig: Tuple[int, int], new: Tuple[int, int]) -> Move:
    r"""
    Get the move from (row1, col1) in \p l1 to (row2, col2) in \p l2.

    :param plyr: Player whose move will be extracted
    :param _orig: Original location to move from
    :param new: New location to move to
    :return: Move corresponding to the move pair
    """
    available_moves = plyr.move_set.avail
    values = list(available_moves.values())
    v = [v for v in values if v.orig == Location(*_orig) and v.new == Location(*new)]
    assert v
    return v[0]


def _verify_num_pieces_and_move_set_size(state: State, num_red_p: int, num_blue_p: int,
                                         num_red_mv: int, num_blue_mv: int):
    r"""
    Verifies the number of pieces and size of the \p MoveSet

    :param state: State of the game
    :param num_red_p: Number of remaining RED pieces
    :param num_blue_p: Number of remaining BLUE pieces
    :param num_red_mv: Number of available moves for RED
    :param num_blue_mv: Number of available moves for BLUE
    """
    # Standardize assert tests
    assert state.red.num_pieces == num_red_p
    assert state.blue.num_pieces == num_blue_p
    assert len(state.red.move_set) == num_red_mv
    assert len(state.blue.move_set) == num_blue_mv


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

    # Verify initial state matches expectations
    _verify_num_pieces_and_move_set_size(state, 7, 7, 4 + 3, 4 + 3)

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

        _verify_num_pieces_and_move_set_size(state, num_red_p, num_blue_p, num_red_mv, num_blue_mv)
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
        _verify_num_pieces_and_move_set_size(state, num_red_p, num_blue_p, num_red_mv, num_blue_mv)


def test_small_direct_attack():
    r""" Test making a direct attack """
    move_list = [(None, None, None, None, 7, 7, 11, 11),
                 (Color.RED, Color.BLUE, (0, 3), (7, 3), 6, 6, 5, 5)
                 ]
    _helper_small_test(move_list)


def test_small_move_then_attack():
    r""" Test making a single move with a scout then a direct attack """
    move_list = [(None, None, None, None, 7, 7, 11, 11),
                 (Color.RED, Color.BLUE, (0, 3), (1, 3), 7, 7, 19, 10),
                 (Color.BLUE, Color.RED, (7, 3), (1, 3), 6, 6, 5, 5)
                 ]
    _helper_small_test(move_list)


def test_single_adjacent_scout():
    r""" Test making a single move with a scout then a direct attack """
    move_list = [(None, None, None, None, 2, 2, 11, 11),
                 (Color.BLUE, Color.BLUE, (2, 4), (2, 3), 1, 1, 0, 0)
                 ]
    _helper_small_test(move_list, state_file="moveset_two_scouts_adjacent.txt")


# noinspection PyProtectedMember
def _helper_small_test(move_info, state_file: str =  "moveset_small_direct_attack.txt"):
    r"""
    Helper function for testing the movements on the small board

    :param move_info: List of move information.  For :math:`n` moves, the length of \p move_info
                      should be :math:`n+1`.  The first element is the initial board configuration.
    """
    path = STATES_PATH / state_file
    assert path.exists(), "Small direct attack state file not found"
    state = State.importer(path, SMALL_BRD)

    _, _, _, _, num_red_p, num_blue_p, num_red_mv, num_blue_mv = move_info[0]
    _verify_num_pieces_and_move_set_size(state, num_red_p, num_blue_p, num_red_mv, num_blue_mv)

    # Test doing moves
    moves, brd = [], [state.write_board()]
    for col, other_col, l1, l2, num_red_p, num_blue_p, num_red_mv, num_blue_mv in move_info[1:]:
        plyr, other = state.get_player(col), state.get_player(other_col)

        m = _get_move_from_player(plyr, l1, l2)
        moves.append(m)
        state.update(moves[-1])
        brd.append([state.write_board()])
        _verify_num_pieces_and_move_set_size(state, num_red_p, num_blue_p,
                                             num_red_mv, num_blue_mv)

    # Test undoing the moves
    for i in range(1, len(moves) - 1):
        assert brd[-i] == state.write_board()

        _, _, l1, l2, num_red_p, num_blue_p, num_red_mv, num_blue_mv = move_info[-i]
        _verify_num_pieces_and_move_set_size(state, num_red_p, num_blue_p,
                                             num_red_mv, num_blue_mv)
        assert moves[-i] == state._stack.top()

        _, _, _, _, num_red_p, num_blue_p, num_red_mv, num_blue_mv = move_info[-i - 1]
        _verify_num_pieces_and_move_set_size(state, num_red_p, num_blue_p,
                                             num_red_mv, num_blue_mv)
        assert brd[-i - 1] == state.write_board()

