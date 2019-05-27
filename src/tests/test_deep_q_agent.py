# -*- coding: utf-8 -*-
r"""
    tests.test_deepq_agent
    ~~~~~~~~~~~~~~~~~~~~~~

    Tests for the \p DeepQAgent class.

    :copyright: (c) 2019 by Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""
import pytest

import torch

import stratego.deep_q_agent as deep_q_agent
from stratego import Location
from stratego.board import Board
from stratego.deep_q_agent import DeepQAgent, ReplayStateTuple
from stratego.move import Move
from stratego.piece import Rank, Color
from stratego.state import State
from stratego.utils import PathOrStr

from testing_utils import NO_BLOCK_BRD, STATES_PATH, STD_BRD


@pytest.fixture(scope="module")
def switch_to_int_tensor():
    r"""
    Helper function to streamline testing to use integer tensors so no floating point errors in
    the test calculations.
    """
    orig_dtype = deep_q_agent.TensorDType
    deep_q_agent.TensorDType = torch.int32
    # Code that will run before your test, for example:
    # A test function will be run at this point
    yield
    # Code that will run after test
    deep_q_agent.TensorDType = orig_dtype


def _make_deep_q_agent(brd: Board = STD_BRD,
                       state_file: PathOrStr = STATES_PATH / "state_move_verify.txt") -> DeepQAgent:
    r""" Helper function used to create a generic Deep Q agent. """
    state = State.importer(state_file, brd)
    return DeepQAgent(brd, state.red, state, disable_import=True)


# noinspection PyProtectedMember
def test_deep_q_constructor():
    r""" Basic constructor related tests """
    agent = _make_deep_q_agent()
    brd = agent._brd  # pylint: disable=protected-access
    num_rows, num_cols = brd.num_rows, brd.num_cols

    # Assuming no blocked squares, scouts can always move the same number of squares
    num_scout_move = 2 * ((num_rows - 1) + (num_cols - 1))
    assert agent._tot_num_scout_moves == num_scout_move
    # Verify the number of output dimensions
    out_count = num_rows * num_cols
    out_count += len(Move.Direction.all())
    out_count += num_scout_move
    assert agent.d_out == out_count


# noinspection PyProtectedMember
# pylint: disable=protected-access
def test_move_to_index_information():
    r"""
    Test that the output node from the neural network has a bijective mapping to possible movements.
    """
    agent = _make_deep_q_agent(NO_BLOCK_BRD)
    state_tuple = ReplayStateTuple(s=agent._state)
    brd = state_tuple.s.board

    # Verify board size as expected
    assert brd.num_loc == brd.num_rows * brd.num_cols

    # Verify non-scout pieces
    p = state_tuple.s.get_player(Color.RED).get_piece_at_loc(Location(0, 0))
    for i in range(state_tuple.s.board.num_loc):
        p.loc = Location(i // brd.num_cols, i % brd.num_cols)
        for j, neighbor in enumerate(p.loc.neighbors()):
            if not brd.is_inside(neighbor): continue
            state_tuple.a = Move(p, p.loc, neighbor)
            board_loc, move_dir_idx = DeepQAgent._get_loc_and_idx_from_move(state_tuple)

            assert board_loc == i
            # Verify neighbor direction matches expectation
            assert j + state_tuple.s.board.num_loc == move_dir_idx

    # Verify scouts specifically
    p._rank = Rank.scout()
    for i in range(state_tuple.s.board.num_loc):
        p.loc = state_tuple.a._orig = Location(i // brd.num_cols, i % brd.num_cols)
        # Left then right
        for sign in [1, -1]:
            ofs = brd.num_loc + (brd.num_rows - 1) + Move.Direction.count()
            if sign == -1:
                ofs += (brd.num_rows - 1) + (brd.num_cols - 1)
            for col_diff in range(0, agent._state.board.num_cols-1):
                new = p.loc.relative(0, sign * (col_diff + 1))
                state_tuple.a._new = new
                board_loc, scout_move_id = DeepQAgent._get_loc_and_idx_from_move(state_tuple)
                assert i == board_loc
                assert ofs + col_diff == scout_move_id
        # Up then down
        for sign in [1, -1]:
            ofs = brd.num_loc + Move.Direction.count()
            if sign == 1:
                ofs += (brd.num_rows - 1) + (brd.num_cols - 1)
            for row_diff in range(0, agent._state.board.num_rows - 1):
                new = p.loc.relative(sign * (row_diff + 1), 0)
                state_tuple.a._new = new
                board_loc, scout_move_id = DeepQAgent._get_loc_and_idx_from_move(state_tuple)
                assert i == board_loc
                assert ofs + row_diff == scout_move_id


# noinspection PyProtectedMember
def test_rank_lookup():
    r""" Simple checks that the rank lookup is reasonable """
    num_rank = len(Rank.get_all())

    # Ensure no duplicate/missing ranks
    keys = DeepQAgent._rank_lookup.keys()
    assert len(keys) == num_rank

    # Verify no duplicate ranks
    vals = set(DeepQAgent._rank_lookup.values())
    assert len(vals) == num_rank
    # Minimum layer number for ranks should start at 0
    assert min(vals) == 0

    # Verify integers from 0 to num_rank - 1
    assert min(vals) == 0 and max(vals) == num_rank - 1
    assert all(isinstance(v, int) for v in vals)

    # Ensure no duplicates for the (non-rank) layers
    other_layers = [DeepQAgent._unk_rank_layer, DeepQAgent._impass_layer]
    assert len(other_layers) == len(set(other_layers))
    # Ensure other layers start right after the rank layers
    assert min(other_layers) == 2 * num_rank
    assert max(other_layers) - min(other_layers) == len(other_layers) - 1


# noinspection PyTypeChecker,PyUnresolvedReferences,PyProtectedMember
@pytest.mark.usefixtures("switch_to_int_tensor")
def test_base_input_builder():
    r""" Verifies that the base input builder has expected contents """
    agent = _make_deep_q_agent()

    base_in = agent._build_base_tensor(agent._brd, agent._d_in)
    # Verify anything but the impass is empty
    base_no_impass = base_in[:, :DeepQAgent._impass_layer]
    sum_impass = base_no_impass.sum()
    assert int(sum_impass) == 0, "Anything not impass should be blank"

    # Check the impass layer is correct
    base_no_impass = base_in.narrow(dim=1, start=DeepQAgent._impass_layer, length=1)
    assert int(base_no_impass.sum()) == len(agent._brd.blocked), "Any loc not impass must be blank"

    # Verify that hte impassable values are in the base tensor
    val = DeepQAgent._IMPASSABLE_VAL
    for loc in agent._brd.blocked:
        assert base_in[0, DeepQAgent._impass_layer, loc.r, loc.c] == val, "Location not blocked"


# noinspection PyTypeChecker,PyUnresolvedReferences,PyProtectedMember
def test_input_builder():
    r""" Verify that the input builder script works as expected """
    agent = _make_deep_q_agent(state_file=STATES_PATH / "deep_q_verify.txt")

    plyrs = [(agent._state.next_player, agent._state.other_player),
             (agent._state.other_player, agent._state.next_player)]
    for plyr, other in plyrs:
        t_in = agent._build_network_input(plyr, other)
        # verify the dimensions
        assert len(t_in.shape) == 4
        # Only a single batch
        assert t_in.size(0) == 1
        assert t_in.size(1) == agent.d_in
        assert t_in.size(2) == agent._brd.num_rows
        assert t_in.size(3) == agent._brd.num_cols

        # Verify the layer count segregation
        plyr_sum = torch.sum(t_in[:, :Rank.count()])
        assert int(plyr_sum) == len(list(plyr.pieces())), "Verify the piece count for the player"
        other_sum = torch.sum(t_in[:, Rank.count():2*Rank.count()])
        assert int(other_sum) == len(list(other.pieces())), "Verify the piece count for the player"

        # noinspection PyUnresolvedReferences
        def _get_in_layer(layer_num: int) -> torch.Tensor:
            return t_in[0, layer_num].view(-1)

        assert plyr.color != other.color
        # assert DeepQAgent._RED_PIECE_VAL != DeepQAgent._BLUE_PIECE_VAL, "Sanity check"

        # Ensure no pieces are unknown (may need to change later)
        assert torch.sum(_get_in_layer(DeepQAgent._unk_rank_layer)) == 0

        # Blocked piece location
        assert torch.sum(_get_in_layer(DeepQAgent._impass_layer)) == len(agent._brd.blocked)


# noinspection PyTypeChecker,PyUnresolvedReferences,PyProtectedMember
def test_null_functions():
    r""" Verify that the input builder script works as expected """
    prev_dtype = deep_q_agent.TensorDType
    deep_q_agent.TensorDType = torch.float32

    agent = _make_deep_q_agent(state_file=STATES_PATH / "deep_q_verify.txt")

    x = agent._build_network_input(agent._state.red, agent._state.blue)
    y = torch.rand((1, agent._d_out))
    y_p = agent._null_invalid_locations(agent._state, x, y.clone())
    assert float(y_p[0, 0]) < DeepQAgent._INVALID_FILL_VAL / 2

    # Verify location of pieces
    piece_locations = {1, 2, 3, 5, agent._state.board.num_cols}
    for i in piece_locations:
        assert float(y_p[0, i]) > -0.1
    # Verification locations where there are no pieces of current player
    for i in range(agent._state.board.num_loc):
        if i in piece_locations: continue
        assert float(y_p[0, i]) < DeepQAgent._INVALID_FILL_VAL / 2

    # Restore data type
    deep_q_agent.TensorDType = prev_dtype
