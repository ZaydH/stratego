# -*- coding: utf-8 -*-
r"""
    agents.deepq_agent
    ~~~~~~~~~~~~~~~~~~

    Deep-Q reinforcement learning agent for Stratego

    :copyright: (c) 2019 by Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""
import copy
import functools
import itertools
import logging
import operator as op
import random
import sys
from typing import Tuple, Iterable, Union
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn
import torch.optim
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from stratego.utils import EXPORT_DIR
from . import Location, Move, utils
from .agent import Agent
from .board import Board
from .piece import Piece, Rank
from .player import Player
from .state import State

# noinspection PyUnresolvedReferences
IS_CUDA = torch.cuda.is_available()
# noinspection PyUnresolvedReferences
TensorDType = torch.float32
ActCls = nn.ReLU


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1) -> nn.Module:
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, groups=groups, bias=False)


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1) -> nn.Module:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def _build_conv_block(in_planes: int, out_planes: int) -> nn.Module:
    r"""
    Constructs a convolutional block

    :param in_planes: Number of input planes/planes
    :param out_planes: Number of output planes/planes
    :return: Convolution block module
    """
    seq = nn.Sequential(_conv3x3(in_planes, out_planes),
                        nn.BatchNorm2d(out_planes),
                        ActCls())
    return seq


def _build_policy_head(in_planes: int, out_planes: int, board_dim: Tuple[int, int],
                       out_dim: int) -> nn.Module:
    r"""
    Construct the policy head that is used to select the next move to perform

    :param in_planes: Number of planes (filters) of the input tensor
    :param out_planes: Number of output planes (filters) from the convolution in this head
    :param board_dim: Dimension of the Stratego board (row x columns)
    :param out_dim: Output dimension of the head (i.e., number of possible moves)
    :return:
    """
    # Number of inputs into the feedforward network
    ff_in_dim = out_planes * functools.reduce(op.mul, board_dim, 1)

    seq = nn.Sequential(_conv1x1(in_planes, out_planes),
                        nn.BatchNorm2d(out_planes),
                        ActCls(),
                        Flatten2Dto1D(),
                        nn.Linear(ff_in_dim, out_dim),
                        # nn.Softmax())
                        nn.Tanh())  # Tanh changed from AlphaGo Zero paper since no value head
    return seq


# def _build_value_head(in_planes: int, out_planes: int, board_dim: Tuple[int, int],
#                       hidden_dim: int = 256) -> nn.Module:
#     r"""
#     Constructs the probabilistic head for computing the value
#
#     :param in_planes: Number of planes (filters) of the input tensor
#     :param out_planes: Number of output planes (filters) from the convolution in this head
#     :param board_dim: Dimension of the Stratego board (row x columns)
#     :param hidden_dim: Number of nodes in the hidden layer in this head
#     :return: Construct the probabilistic head
#     """
#     # Number of inputs into the feedforward network
#     ff_in_dim = out_planes * functools.reduce(op.mul, board_dim, 1)
#
#     seq = nn.Sequential(_conv1x1(in_planes, out_planes),
#                         nn.BatchNorm2d(out_planes),
#                         ActCls(),
#                         Flatten2Dto1D(),
#                         nn.Linear(ff_in_dim, hidden_dim),
#                         ActCls(),
#                         nn.Linear(hidden_dim, 1),
#                         nn.Tanh())
#     return seq


class Flatten2Dto1D(nn.Module):
    r""" Helper layer that takes a 2D tensor and flattens it to a 1D vector """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # noinspection PyUnresolvedReferences
        return x.view((x.shape[0], -1))


class ResBlock(nn.Module):
    r"""
    Residual block with the following structure:

    (1) Convolution kernel size 3x3, stride 1
    (2) Batch normalization
    (3) Non-linearity
    (4) Convolution kernel size 3x3, stride 1
    (5) Batch normalization
    (6) Skip connection adding input to output of step 5
    (7) Non-linearity
    """
    NUM_PLANES = 256  # Filters

    def __init__(self):
        super().__init__()
        self._seq = nn.Sequential(_conv3x3(self.NUM_PLANES, self.NUM_PLANES),
                                  nn.BatchNorm2d(self.NUM_PLANES),
                                  ActCls(),
                                  _conv3x3(self.NUM_PLANES, self.NUM_PLANES),
                                  nn.BatchNorm2d(self.NUM_PLANES)
                                  )
        self._final_act = ActCls()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r""" Forward pass through the network """
        y = self._seq(x)
        # noinspection PyUnresolvedReferences
        y = x + y
        return self._final_act(y)


@dataclass(init=True)
class ReplayStateTuple:
    s: State = None
    a: Move = None
    base_tensor: torch.Tensor = None
    r: torch.Tensor = None

    def is_terminal(self) -> bool:
        r""" Returns True if this state is terminal """
        return self.r in DeepQAgent.TERMINAL_REWARDS


class ReplayMemory:
    r""" Deep Q replay memory :math:`\mathcal{D}` as described in:

    Mnih et al. "Playing Atari with Deep Reinforcement Learning." (2013).
    """
    _N = 1000

    def __init__(self):
        self._buf = []
        self._next = 0

    def add(self, s: ReplayStateTuple) -> None:
        r"""
        Adds the state \p s to the ReplayMemory.  After the memory reaches size :math:`N`, then
        replay memory acts like a circular buffer.

        Object \s is NOT stored in the replay memory.  Instead a deepcopy is stored to prevent
        conflicts with the learner.

        :param s: State to add to the replay memory
        """
        s = copy.deepcopy(s)
        if len(self._buf) < self._N:
            self._buf.append(s)
            return
        self._buf[self._next] = s
        self._next += 1
        if self._next == self._N: self._next = 0

    def get_random(self) -> ReplayStateTuple:
        r""" Select a random element from the replay memory """
        return random.choice(self._buf)


class DeepQAgent(Agent, nn.Module):
    r""" Agent that encapsulates the Deep-Q algorithm described in papers such as:
    Mnih et al. "Playing Atari with Deep Reinforcement Learning." (2013).
    """
    _PIECE_VAL = 1.
    # _RED_PIECE_VAL = 1
    # _BLUE_PIECE_VAL = -1
    _IMPASSABLE_VAL = 1

    _NUM_RES_BLOCKS = 3

    # Training parameters
    _M = 100000
    _T = 4000  # Maximum number of moves for a state
    _EPS_START = 0.25
    _gamma = 0.98
    _lr = 1e-4
    _f_loss = nn.MSELoss()
    _INVALID_MOVE_REWARD = -torch.ones((1, 1))  # Must be -1 since output is tanh
    _LOSS_REWARD = -torch.ones_like(_INVALID_MOVE_REWARD)
    _WIN_REWARD = torch.ones_like(_LOSS_REWARD)

    TERMINAL_REWARDS = (_LOSS_REWARD, _WIN_REWARD, _INVALID_MOVE_REWARD)

    # Converts rank to a layer
    _rank_lookup = {r: i for i, r in enumerate(Rank.get_all())}
    # Layer for a piece whose rank is unknown
    _unk_rank_layer = 2 * Rank.count()
    # Layer where impassable locations are marked
    _impass_layer = _unk_rank_layer + 1
    # # Layer indicating whose turn is next
    # _next_turn_layer = _impass_layer + 1

    _EXPORTED_MODEL = EXPORT_DIR / "final_deep_q.pth"

    def __init__(self, brd: Board, plyr: Player, other: Player, eps_end: float = 1e-4,
                 disable_import: bool = False):
        r"""
        :param brd: Game board
        :param plyr: Player who will be controlled by the agent.
        :param other: Other player, i.e., the one not controlled by this object
        """
        if plyr.color == other.color: raise ValueError("Two player colors cannot match")

        nn.Module.__init__(self)
        Agent.__init__(self, plyr)

        self._other = other
        self._brd = brd
        # Layer 0 to d-1: 1 if piece of i^th rank is present, otherwise 0
        # Layer d: 1 for any piece of unknown rank (not used)
        # Layer d+1: 1 for an impassable location, 0 otherwise
        # Layer d+2: 1 for which player's turn is next, 0 otherwise
        self._d_in = 2 * Rank.count() + 2

        self._EPS_END = eps_end

        # Maximum Size of the Policy Set
        # From each location on the board, the player could (assuming no blocked spaces) move to any
        # location in the same column or row. This defines the dimensions of the output.
        self._d_out = self._brd.num_rows * self._brd.num_cols * self._num_scout_moves()

        # Tensor used as the basis for move tensors
        self._base_in = DeepQAgent._build_base_tensor(self._brd, self.d_in)

        self._construct_network()
        self._replay = None
        if not disable_import and DeepQAgent._EXPORTED_MODEL.exists():
            msg = "Importing saved model: \"%s\"" % str(DeepQAgent._EXPORTED_MODEL)
            logging.debug("Starting: %s" % msg)
            utils.load_module(self, DeepQAgent._EXPORTED_MODEL)
            logging.debug("COMPLETED: %s" % msg)
        # Must be last in constructor to ensure proper CUDA enabling
        if IS_CUDA: self.cuda()

    @property
    def d_in(self) -> int:
        r""" Number of input layers input into the agent's """
        return self._d_in

    @property
    def d_out(self) -> int:
        r""" Accessor for the maximum size of the policy set """
        return self._d_out

    def train_network(self, s_0: State):
        Move.DISABLE_ASSERT_CHECKS = True
        self._replay = ReplayMemory()
        optim = torch.optim.Adam(self.parameters(), lr=self._lr)
        eps_range = np.linspace(self._EPS_START, self._EPS_END, num=self._M, endpoint=True)
        # Decrement epsilon with each step
        for episode, self._eps in zip(range(1, self._M + 1), eps_range):
            # noinspection PyProtectedMember
            t = ReplayStateTuple(s=copy.deepcopy(s_0),
                                 base_tensor=DeepQAgent._build_base_tensor(s_0.board, self.d_in))

            f_out = open("deep-q_train_moves.txt", "w+")
            f_out.write("# PlayerColor,StartLoc,EndLoc")

            logging.info("Starting episode %d of %d", episode, self._M)
            progress_bar = tqdm(range(self._T), total=self._T, file=sys.stdout)
            for _ in progress_bar:
                # With probability < \epsilon, select a random action
                if not t.s.next_player.move_set.is_empty():
                    if random.random() < self._eps:
                        t.a = t.s.next_player.get_random_move()
                    # Select (epsilon-)greedy action
                    else:
                        _, policy, t.a = self._get_state_move(t)
                        # Treat illegal moves as an instant loss
                        if not t.s.next_player.is_valid_next(t.a):
                            t.r = self._INVALID_MOVE_REWARD
                            self._punish_invalid_move(policy, optim)
                            continue
                    f_out.write("\n%s,%s,%s" % (t.a.piece.color.name, str(t.a.orig), str(t.a.new)))
                    f_out.flush()

                # Handle case player already lost
                if t.s.is_game_over():
                    t.r = self._LOSS_REWARD
                # Player about to win
                elif t.a.is_attack() and t.a.attacked.rank == Rank.flag():
                    t.r = self._WIN_REWARD
                # Game not over yet
                else:
                    t.r = torch.zeros_like(self._WIN_REWARD)
                self._replay.add(t)

                j = self._replay.get_random()
                y_j = j.r
                if not j.is_terminal():
                    j.s.update(j.a)
                    if j.s.next_player.move_set.is_empty():
                        y_j = -DeepQAgent._LOSS_REWARD
                    else:
                        # ToDo Need to fix how board state measured since player changed after move
                        _, y_j_1_val, _ = self._get_state_move(j)
                        y_j = y_j - self._gamma * y_j_1_val
                        # ToDo may need to rollback multiple moves
                    j.s.rollback()

                q_j = self._get_action_output_score(j)
                loss = self._f_loss(q_j, y_j)
                optim.zero_grad()
                loss.backward()
                optim.step()

                # Advance to next state
                if t.s.is_game_over():
                    progress_bar.close()
                    break
                t.s.update(t.a)
            f_out.close()
            utils.save_module(self, EXPORT_DIR / ("episode_%04d.pth" % episode))
            logging.info("COMPLETED episode %d of %d", episode, self._M)
        utils.save_module(self, DeepQAgent._EXPORTED_MODEL)
        Move.DISABLE_ASSERT_CHECKS = False

    def _punish_invalid_move(self, max_tensor: torch.Tensor, optim: Optimizer) -> None:
        r"""
        Updates the network to punish for illegal moves

        :param max_tensor: Max tensor from the learner
        :param optim: Optimizer
        """
        loss = self._f_loss(max_tensor, DeepQAgent._INVALID_MOVE_REWARD)
        optim.zero_grad()
        loss.backward()
        optim.step()

    def _num_scout_moves(self) -> int:
        r"""
        Maximum number of moves a scout can perform. This function deliberately does not consider
        the case where some number of moves may be blocked due an obstruction within the board
        (i.e, not the board edge).
        """
        return (self._brd.num_cols - 1) + (self._brd.num_rows - 1)

    def _get_state_move(self, state_tuple: ReplayStateTuple) -> Tuple[int, torch.Tensor, Move]:
        r"""
        Gets the move corresponding to the specified state

        :param state_tuple: Deep-Q learning state
        :return: Tuple of the output node, maximum valued error, and the selected move respectively
        """
        x = DeepQAgent._build_input_tensor(state_tuple.base_tensor, state_tuple.s.pieces(),
                                           state_tuple.s.next_player)
        policy = self.forward(x)
        output_node = int(torch.argmax(policy))
        move = self._convert_output_node_to_move(output_node, state_tuple.s.next_player,
                                                 state_tuple.s.other_player)
        # return output_node, move
        max_tensor, _ = policy.max(dim=1, keepdim=True)
        return output_node, max_tensor, move

    def _get_move_from_output_node(self, output_node: Union[torch.Tensor, int],
                                   plyr: Player, other: Player) -> Move:
        r"""
        Gets the move for the current player based on the node with the highest value

        :param output_node: Index of the output node
        """
        return self._convert_output_node_to_move(output_node, plyr, other)

    def _get_action_output_score(self, state_tuple: ReplayStateTuple) -> torch.Tensor:
        r"""
        Used in training the agent.  Returns the output score for action :math:`a_j` in :math:`s_j`

        :param state_tuple: State at step :math:`j`
        :return:
        """
        output_node = self._get_output_node_from_move(state_tuple.a)
        x = self._build_input_tensor(state_tuple.base_tensor, state_tuple.s.pieces(),
                                     state_tuple.s.next_player)
        y = self.forward(x)
        return y[:, output_node:output_node + 1]

    def _convert_output_node_to_move(self, output_node: Union[torch.Tensor, int], plyr: Player,
                                     other: Player) -> Move:
        r"""
        Creates the \p Move object corresponding to the specified \p output_node for \p plyr
        and \p other.

        :param output_node: Output node number for the network
        :param plyr: Player making the move
        :param other: Player NOT moving this round
        :return: Move corresponding to specified output node
        """
        assert plyr.color != other.color, "Player colors must differ"
        # Handle case of input tensor for streamlining the implementation
        if isinstance(output_node, torch.Tensor):
            assert all(x == 1 for x in output_node.shape), "More than one element in output_node"
            output_node = int(output_node)
        orig, new = self._get_locs_from_output_node(output_node)
        p = plyr.get_piece_at_loc(orig)
        attacked = other.get_piece_at_loc(new)
        return Move(p, orig, new, attacked)

    def _get_output_node_from_move(self, m: Move) -> int:
        r"""
        Converts a move to the corresponding output node of the policy network.

        :param m: Move on the board
        :return: Converts a move into the corresponding node in the output layer of the policy
                 network
        """
        output_node = self._num_scout_moves() * (m.orig.r * self._brd.num_cols + m.orig.c)

        if m.orig.c - m.new.c != 0:
            output_node += m.new.c - (1 if m.new.c > m.orig.c else 0)
        else:
            output_node += self._brd.num_cols - 1  # Maximum number of horizontal moves
            output_node += m.new.r - (1 if m.new.r > m.orig.r else 0)

        return output_node

    def _get_locs_from_output_node(self, output_node: int) -> Tuple[Location, Location]:
        r"""
        Converts the output node number to a move from the original location to the new location.
        The basic idea of the function is that from a given node, there are

        :math:`(num_rows - 1) + (num_cols - 1)`

        possible moves.  This comes from how scouts move.  Therefore, each location has a constant
        number of possible moves (call that number :math:`n`) that can originate at that location.
        Therefore, we can partition the output node set into :math:`\#rows \cdot \#cols` disjoint
        subsets of size n.  This is how the original (source) \p Location is calculated.

        The destination location is ordered by all possible horizontal moves first (starting at
        column 0) and then the vertical moves next.

        Separated for improved testability.

        :param output_node: Identification of the output node selected to play a move
        :return: Tuple of the original and new location of the move respectively.
        """
        if output_node < 0 or output_node >= self.d_out:
            raise ValueError("output_node must be in set {0,...,d_out - 1}")

        source_loc = output_node // self._num_scout_moves()
        orig = Location(source_loc // self._brd.num_cols, source_loc % self._brd.num_cols)

        # Calculate the destination location based on the (num_rows + num_cols - 2) choices
        dest_loc = output_node % self._num_scout_moves()
        row_off = col_off = 0
        max_num_horz_moves = self._brd.num_cols - 1
        if dest_loc < max_num_horz_moves:
            col_off = (dest_loc - orig.c) + (1 if dest_loc >= orig.c else 0)
        else:
            dest_loc -= max_num_horz_moves
            row_off = (dest_loc - orig.r) + (1 if dest_loc >= orig.r else 0)
        new = orig.relative(row_off, col_off)
        return orig, new

    def _construct_network(self):
        r""" Constructs the neural network portions of the agent. """
        self._q_net = nn.Sequential(_build_conv_block(self.d_in, ResBlock.NUM_PLANES))
        for i in range(self._NUM_RES_BLOCKS):
            self._q_net.add_module("Res_%02d" % i, ResBlock())

        self._head_policy = _build_policy_head(ResBlock.NUM_PLANES, out_planes=1,
                                               board_dim=self._brd.dim(), out_dim=self.d_out)
        # self._head_value = _build_value_head(ResBlock.NUM_PLANES, out_planes=1,
        #                                      board_dim=self._brd.dim())

    def get_next_move(self) -> Move:
        r"""
        Select the next move to be played.
        :return: Move to be performed
        """
        x = self._build_network_input(self._plyr, self._other)
        # Gradients no needed since will not be push backwards
        # noinspection PyUnresolvedReferences
        with torch.no_grad():
            policy = self.forward(x)
        while True:
            out_node = torch.argmax(policy, dim=1)
            try:
                m = self._get_move_from_output_node(out_node, self._plyr, self._other)
                # As a safety always ensure selected move is valid
                if m in self._plyr.move_set:
                    return m
                logging.warning("Tried to select move %s to %s for color %s but move is invalid",
                                m.orig, m.new, self.color)
            except (ValueError, AssertionError):
                logging.warning("Runtime exception with out_node: %d" % out_node)
            # Zero out that move and select a different one
            policy[:, out_node] = -torch.ones((policy.size(0)), dtype=TensorDType)

    # def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     r"""
    #     Passes the input tensor \p x through the Q network
    #
    #     :param x: Input tensor
    #     :return: Tuple of the ten
    #     """
    #     # noinspection PyUnresolvedReferences
    #     assert x.shape[1:] == [self.d_in, self._brd.num_rows, self._brd.num_cols]
    #     y = self._q_net(x)
    #     return self._head_policy(y), self._head_value(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Passes the input tensor \p x through the Q network

        :param x: Input tensor
        :return: Tuple of the ten
        """
        # noinspection PyUnresolvedReferences
        assert x.shape[1:] == torch.Size([self.d_in, self._brd.num_rows, self._brd.num_cols])
        y = self._q_net(x)
        return self._head_policy(y)

    def _build_network_input(self, plyr: Player, other: Player) -> torch.Tensor:
        r"""
        Constructs the tensor to input into the Q-network.

        :return: Tensor to put into the network
        """
        pieces = itertools.chain(plyr.pieces(), other.pieces())
        return self._build_input_tensor(self._base_in, pieces, plyr)

    @staticmethod
    def _build_base_tensor(brd: Board, d_in: int) -> TensorDType:
        r"""
        Constructs a base tensor for a board

        :param brd: \p Board object upon which the base tensor is based
        :param d_in: Number of layers in the \p torch tensor to be input into the deep-Q network
        :return: Base tensor based on the board
        """
        base_in = torch.zeros((1, d_in, brd.num_rows, brd.num_cols), dtype=TensorDType)
        # Mark the impassable locations on the board
        for l in brd.blocked:
            # noinspection PyUnresolvedReferences
            base_in[0, DeepQAgent._impass_layer, l.r, l.c] = DeepQAgent._IMPASSABLE_VAL
        return base_in

    # noinspection PyUnresolvedReferences
    @staticmethod
    def _build_input_tensor(base_in: torch.Tensor, pieces: Iterable[Piece], next_player: Player):
        r"""
        Constructs the tensor to input into the deep Q network.

        :param base_in: Base tensor to which the piece information will be added.  Base tensor
                        has
        :param pieces: Iterable set of all active pieces
        :return: Tensor that can be input into the network
        """
        x = base_in.clone()
        for p in pieces:
            layer_num = 2 * DeepQAgent._rank_lookup[p.rank]
            if p.color != next_player.color:
                layer_num += 1
            x[0, layer_num, p.loc.r, p.loc.c] = DeepQAgent._PIECE_VAL
            # if p.color == Color.RED: color_val = DeepQAgent._RED_PIECE_VAL
            # else: color_val = DeepQAgent._BLUE_PIECE_VAL
            #
            # # Mark piece as present
            # x[0, DeepQAgent._rank_lookup[p.rank], p.loc.r, p.loc.c] = color_val
            # if p.color == next_player.color:
            #     x[0, DeepQAgent._next_turn_layer, p.loc.r, p.loc.c] = color_val
            # ToDo Need to update when imperfect information
        return x
