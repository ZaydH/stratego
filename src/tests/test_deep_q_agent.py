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

from stratego.board import Board
from stratego.deep_q_agent import DeepQAgent
from stratego.location import Location
from stratego.move import Move
from stratego.piece import Color, Rank, Piece
from stratego.state import State
from stratego.utils import PathOrStr

from testing_utils import NO_BLOCK_BRD, STATES_PATH, STD_BRD


def _make_deep_q_agent(brd: Board = STD_BRD,
                       state_file: PathOrStr = STATES_PATH / "state_move_verify.txt") -> DeepQAgent:
    r""" Helper function used to create a generic Deep Q agent. """
    state = State.importer(state_file, brd)
    return DeepQAgent(brd, state.red, state.blue, disable_import=True)


# noinspection PyProtectedMember
def test_deep_q_constructor():
    r""" Basic constructor related tests """
    agent = _make_deep_q_agent()
    brd = agent._brd  # pylint: disable=protected-access
    num_rows, num_cols = brd.num_rows, brd.num_cols

    # Assuming no blocked squares, scouts can always move the same number of squares
    num_scout_move = agent._num_scout_moves()  # pylint: disable=protected-access
    assert num_scout_move == (num_rows - 1) + (num_cols - 1)
    # Verify the number of output dimensions
    assert agent.d_out == num_rows * num_cols * num_scout_move


# noinspection PyProtectedMember
# pylint: disable=protected-access
def test_output_node_to_locs():
    r"""
    Test that the output node from the neural network has a bijective mapping to possible movements.
    """
    agent = _make_deep_q_agent(NO_BLOCK_BRD)

    # Duplicate but sanity check dimensions for below
    brd = agent._brd  # pylint: disable=protected-access
    num_rows, num_cols = brd.num_rows, brd.num_cols
    num_scout_move = agent._num_scout_moves()  # pylint: disable=protected-access
    assert num_scout_move == (num_rows - 1) + (num_cols - 1)

    # Ensure boundary checks
    with pytest.raises(ValueError):
        agent._get_locs_from_output_node(-1)

    # Ensure bijective mapping of output nodes to moves
    all_moves = set()
    for i in range(agent.d_out):
        all_moves.add(agent._get_locs_from_output_node(i))
    assert len(all_moves) == agent.d_out, "Multiple output nodes map to same move"

    assert Location(0, 0), Location(0, 1) == agent._get_locs_from_output_node(0)
    assert Location(0, 1), Location(0, 0) == agent._get_locs_from_output_node(num_scout_move)

    # Test the conversion with the inverse
    for i in range(agent.d_out):
        orig, new = agent._get_locs_from_output_node(i)
        p = Piece(Color.RED, Rank.scout(), orig)
        m = Move(p, orig, new)
        assert agent._get_output_node_from_move(m) == i


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
    other_layers = [DeepQAgent._unk_rank_layer, DeepQAgent._impass_layer,
                    DeepQAgent._next_turn_layer]
    assert len(other_layers) == len(set(other_layers))
    # Ensure other layers start right after the rank layers
    assert min(other_layers) == num_rank
    assert max(other_layers) - min(other_layers) == len(other_layers) - 1


# noinspection PyTypeChecker,PyUnresolvedReferences,PyProtectedMember
def test_input_builder():
    r""" Verify that the input builder script works as expected """
    agent = _make_deep_q_agent()

    t_in = agent._build_network_input(agent._plyr, agent._other)
    # verify the dimensions
    assert len(t_in.shape) == 4
    # Only a single batch
    assert t_in.size(0) == 1
    assert t_in.size(1) == agent.d_in
    assert t_in.size(2) == agent._brd.num_rows
    assert t_in.size(3) == agent._brd.num_cols

    # noinspection PyUnresolvedReferences
    def _get_in_layer(layer_num: int) -> torch.Tensor:
        return t_in[0, layer_num].view(-1)

    num_plyr_piece = len(list(agent._plyr.pieces()))
    num_other_piece = len(list(agent._other.pieces()))
    _color_layers = t_in[0, 0:len(Rank.get_all())].view([-1])
    # Assert the number of pieces is correct
    assert DeepQAgent._RED_PIECE_VAL != DeepQAgent._BLUE_PIECE_VAL, "Sanity check"
    num_red = sum(_color_layers == DeepQAgent._RED_PIECE_VAL)
    assert num_red == num_plyr_piece
    num_blue = sum(_color_layers == DeepQAgent._BLUE_PIECE_VAL)
    assert num_blue == num_other_piece

    # Ensure no pieces are unknown (may need to change later)
    assert torch.sum(_get_in_layer(DeepQAgent._unk_rank_layer)) == 0

    # Blocked piece location
    assert torch.sum(_get_in_layer(DeepQAgent._impass_layer)) == len(agent._brd.blocked)
