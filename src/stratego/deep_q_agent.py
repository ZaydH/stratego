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
import pickle
import random
from pathlib import Path
from typing import Tuple, Iterable, Union, List, Optional
from dataclasses import dataclass

import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim

from stratego.random_agent import RandomAgent
from . import Location, Move, utils, Game
from .agent import Agent
from .board import Board, ToEdgeLists
from .piece import Piece, Rank
from .player import Player
from .state import State
from .utils import EXPORT_DIR, SERIALIZE_DIR, PathOrStr

IS_CUDA = torch.cuda.is_available()
TensorDType = torch.float32
if IS_CUDA:
    device = torch.device('cuda:0')
    # noinspection PyUnresolvedReferences
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
ActCls = nn.LeakyReLU


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
    seq = nn.Sequential(_conv1x1(in_planes, out_planes),
                        nn.BatchNorm2d(out_planes),
                        ActCls(),
                        Flatten2Dto1D())

    # Number of inputs into the feedforward network
    num_hidden_layers = 2
    hidden_dim = 128
    prev_in_dim = out_planes * functools.reduce(op.mul, board_dim, 1)
    for i in range(num_hidden_layers):
        seq.add_module("FF_Lin_%02d" % i, nn.Linear(prev_in_dim, hidden_dim))
        seq.add_module("FF_Act_%02d" % i, ActCls())
        prev_in_dim = hidden_dim

    seq.add_module("FF_Lin_OUT", nn.Linear(prev_in_dim, out_dim))
    # seq.add_module("output", nn.Softmax(dim=1))  # Softmax across columns
    seq.add_module("output", nn.Tanh())  # Tanh changed from AlphaGo Zero paper since no value head
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
    def forward(self, x: Tensor) -> Tensor:
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
    # NUM_PLANES = 256  # Filters
    NUM_PLANES = 32  # Filters

    def __init__(self):
        super().__init__()
        self._seq = nn.Sequential(_conv3x3(self.NUM_PLANES, self.NUM_PLANES),
                                  nn.BatchNorm2d(self.NUM_PLANES),
                                  ActCls(),
                                  _conv3x3(self.NUM_PLANES, self.NUM_PLANES),
                                  nn.BatchNorm2d(self.NUM_PLANES)
                                  )
        self._final_act = ActCls()

    def forward(self, x: Tensor) -> Tensor:
        r""" Forward pass through the network """
        y = self._seq(x)
        # noinspection PyUnresolvedReferences
        y = x + y
        return self._final_act(y)


@dataclass(init=True)
class ReplayStateTuple:
    s: State = None
    s_p: State = None
    a: Move = None
    base_tensor: Tensor = None
    r: Tensor = None

    _next_move_empty: bool = None

    def compress_movestack(self) -> None:
        r""" Compress the MoveStack to save space and make copies faster """
        self.s.compress_movestack()

    def is_terminal(self) -> bool:
        r""" Returns True if this state is terminal """
        for x in DeepQAgent.TERMINAL_REWARDS:
            if self.r.equal(x): return True
        return False

    def create_s_p(self) -> None:
        r""" Create the successor state if needed """
        if self.s_p is not None: return

        self.s_p = copy.deepcopy(self.s)
        self.s_p.update(self.a)
        # Empty move stack if cyclic moves may make it appear as there are no moves
        if self.s_p.is_next_moves_unavailable(): self.s_p.empty_movestack()


class ReplayMemory:
    r""" Deep Q replay memory :math:`\mathcal{D}` as described in:

    Mnih et al. "Playing Atari with Deep Reinforcement Learning." (2013).
    """
    _N = 1000000

    def __init__(self):
        self._buf = []  # type: List[ReplayStateTuple]
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

    def get_random(self, batch_size=32) -> List[ReplayStateTuple]:
    # def get_random(self, batch_size=8) -> List[ReplayStateTuple]:
        r""" Select a random element from the replay memory """
        if len(self._buf) < batch_size:
            return self._buf
        return np.random.choice(self._buf, size=batch_size, replace=False)

    def make_all_s_p(self) -> None:
        r"""
        Pre-create the s' states for all elements in the buffer.  This is useful when creating the
        random pre-move buffer.
        """
        msg = "Creating all replay buffer s_p objects"
        logging.info("Starting: %s" % msg)
        for x in self._buf:
            x.create_s_p()
        logging.info("COMPLETED: %s" % msg)

    def __len__(self) -> int:
        return len(self._buf)


class DeepQAgent(Agent, nn.Module):
    r"""
    Agent that encapsulates the Deep-Q algorithm described in papers such as: Mnih et al. "Playing
    Atari with Deep Reinforcement Learning." (2013).
    """
    _PIECE_VAL = 1.
    # _RED_PIECE_VAL = 1
    # _BLUE_PIECE_VAL = -1
    _IMPASSABLE_VAL = 1.

    # _NUM_RES_BLOCKS = 9
    _NUM_RES_BLOCKS = 4

    # Training parameters
    _M = 10000
    _T = 1500  # Maximum number of moves for a state
    _EPS_START = 0.5
    _gamma = 0.98
    # _LR_START = 0.2
    _LR_START = 1E-3
    _f_loss = nn.MSELoss()
    _LOSS_REWARD = -torch.ones((1, 2))  # Must be -1 since output is tanh
    # _INVALID_MOVE_REWARD = _LOSS_REWARD
    _NON_TERMINAL_MOVE_REWARD = torch.full_like(_LOSS_REWARD, -3E-3)
    _WIN_REWARD = torch.ones_like(_LOSS_REWARD)

    _MAX_NUM_CONSECUTIVE_MOVES = 10

    # TERMINAL_REWARDS = (_LOSS_REWARD, _WIN_REWARD, _INVALID_MOVE_REWARD)
    TERMINAL_REWARDS = (_LOSS_REWARD, _WIN_REWARD)

    # _INVALID_FILL_VAL = 10 * float(_INVALID_MOVE_REWARD[0, 0])
    _INVALID_FILL_VAL = 10 * float(_LOSS_REWARD[0, 0])

    # _CHECKPOINT_EPISODE_FREQUENCY = 100
    _CHECKPOINT_EPISODE_FREQUENCY = 10
    _NUM_HEAD_TO_HEAD_GAMES = 101
    # _NUM_HEAD_TO_HEAD_GAMES = 5
    _TRAIN_BEST_MODEL = EXPORT_DIR / "_checkpoint_best_model.pth"

    # Converts rank to a layer
    _rank_lookup = {r: i for i, r in enumerate(Rank.get_all())}
    # Layer for a piece whose rank is unknown
    _unk_rank_layer = 2 * Rank.count()
    # Layer where impassable locations are marked
    _impass_layer = _unk_rank_layer + 1
    # # Layer indicating whose turn is next
    # _next_turn_layer = _impass_layer + 1

    _EXPORTED_MODEL = EXPORT_DIR / "final_deep_q.pth"

    def __init__(self, brd: Board, plyr: Player, state: State, eps_end: float = 1e-4,
                 disable_import: bool = False):
        r"""
        :param brd: Game board
        :param plyr: Player who will be controlled by the agent.
        :param disable_import: Disable importing of an agent from disk
        """
        # if plyr.color == other.color: raise ValueError("Two player colors cannot match")

        nn.Module.__init__(self)
        Agent.__init__(self, plyr)

        self._state = state
        # self._other = other
        self._brd = brd
        # Layer 0 to d-1: 1 if piece of i^th rank is present, otherwise 0
        # Layer d: 1 for any piece of unknown rank (not used)
        # Layer d+1: 1 for an impassable location, 0 otherwise
        # Layer d+2: 1 for which player's turn is next, 0 otherwise
        self._d_in = 2 * Rank.count() + 2

        self._eps = self._EPS_START
        self._EPS_END = eps_end

        self._set_initial_out_dim()

        # Tensor used as the basis for move tensors
        self._base_in = DeepQAgent._build_base_tensor(self._brd, self.d_in)

        # Add feature for making a random move with some probability
        self._make_rand_move_prob = None

        self._construct_network()
        self._replay = None  # type: Optional[ReplayMemory]

        self._episode = 0
        self._optim = optim.Adam(self.parameters(), lr=DeepQAgent._LR_START)
        self._lr_sched = optim.lr_scheduler.MultiStepLR(self._optim, milestones=[2000, 5000],
                                                        gamma=0.1)

        for model_path in [DeepQAgent._EXPORTED_MODEL, DeepQAgent._TRAIN_BEST_MODEL]:
            if model_path.exists(): break
        if not disable_import and model_path.exists():
            # msg = "Importing saved model: \"%s\"" % str(model_path)
            # logging.debug("Starting: %s" % msg)
            utils.load_module(self, model_path)
            # logging.debug("COMPLETED: %s" % msg)
        # Must be last in constructor to ensure proper CUDA enabling
        if IS_CUDA: self.cuda()

    def _set_nn_param(self, name: str, val: Union[int, float]):
        r"""
        Helper function to standardize setting parameters in the \p Module.  These parameters will
        be saved even after a saving of the state dictionary.

        :param name: Name of the parameter to set
        :param val: Value to set the parameter \p name
        """
        # noinspection PyUnresolvedReferences
        tensor_dtype = torch.int32 if isinstance(val, int) else torch.float32
        tensor_val = torch.tensor([val], dtype=tensor_dtype)
        self.__setattr__(name, nn.Parameter(tensor_val, requires_grad=False))

    def _get_nn_param(self, param_name: str) -> Union[int, float]:
        r"""
        Helper function to standardize setting parameters in the \p Module.  These parameters will
        be saved even after a saving of the state dictionary.
        """
        param_val = self.__getattr__(param_name)
        # noinspection PyUnresolvedReferences
        param_type = int if param_val.dtype == torch.int32 else float
        return param_type(param_val[0])

    @property
    def _eps(self) -> float:
        r""" Accessor for training loss function parameter :math:`\epsilon` """
        return self._get_nn_param("_epsilon")

    @_eps.setter
    def _eps(self, val: float) -> None:
        r""" MUTATOR for training loss parameter :math:`\epsilon` """
        self._set_nn_param("_epsilon", val)

    @property
    def _episode(self) -> int:
        r""" Accessor for the current EPISODE number """
        return self._get_nn_param("_episode_param")

    @_episode.setter
    def _episode(self, val: int) -> None:
        r""" MUTATOR for the current EPISODE number """
        self._set_nn_param("_episode_param", val)

    @property
    def d_in(self) -> int:
        r""" Number of input layers input into the agent's """
        return self._d_in

    @property
    def d_out(self) -> int:
        r""" Accessor for the maximum size of the policy set """
        return self._d_out

    def _set_initial_out_dim(self) -> None:
        r"""
        New network architecture has three parts.  Piece location, movement direction, and scout
        distance.  Clearly the last component only applies to scouts.
        """
        self._tot_num_scout_moves = 2 * (self._brd.num_rows - 1) + 2 * (self._brd.num_cols - 1)

        self._d_out = self._brd.num_loc + Move.Direction.count() + self._tot_num_scout_moves

        # # Maximum Size of the Policy Set
        # # From each location on the board, the player could (assuming no blocked spaces) move to
        # # any location in the same column or row. This defines the dimensions of the output.
        # self._d_out = self._brd.num_rows * self._brd.num_cols * self._num_scout_moves()

    def train_network(self, s_0: State):
        r""" Train the agent """
        s_0.is_training = True

        # If a trained model exists, load it. Otherwise, backup the default model
        if DeepQAgent._TRAIN_BEST_MODEL.exists():
            utils.load_module(self, DeepQAgent._TRAIN_BEST_MODEL)
        else:
            utils.save_module(self, DeepQAgent._TRAIN_BEST_MODEL)

        self.train()
        self._replay = ReplayMemory()
        self._fill_initial_move_buffer(copy.deepcopy(s_0))

        num_rem_episodes = self._M - self._episode
        eps_range = np.linspace(self._eps, self._EPS_END, num=num_rem_episodes, endpoint=True)
        # Decrement epsilon with each step
        for self._episode, self._eps in zip(range(self._episode+1, self._M + 1), eps_range):
            logging.debug("Episode %d: epsilon = %.6f", self._episode, self._eps)
            # noinspection PyProtectedMember
            t = ReplayStateTuple(s=copy.deepcopy(s_0),
                                 base_tensor=DeepQAgent._build_base_tensor(s_0.board, self.d_in))
            self._train_episode(t)
            # noinspection PyArgumentList
            self._lr_sched.step()

            if self._episode % DeepQAgent._CHECKPOINT_EPISODE_FREQUENCY == 0:
                self._compare_head_to_head(s_0)

        utils.save_module(self, DeepQAgent._EXPORTED_MODEL)
        Move.DISABLE_ASSERT_CHECKS = False

    def _train_episode(self, t: ReplayStateTuple) -> None:
        r"""
        Performs training for a single epoch

        :param t: Initial state tuple for the episode
        """
        f_out = open("deep-q_train_moves.txt", "w+")
        f_out.write("# PlayerColor,StartLoc,EndLoc")

        tot_num_moves = num_rand_moves = 0
        logging.info("Starting episode %d of %d", self._episode, self._M)
        while True:
            # ============================== #
            #          Advance Step          #
            # ============================== #
            t, moves_in_round, rand_moves_in_round = self._play_moves(t, f_out)
            tot_num_moves += moves_in_round
            num_rand_moves += rand_moves_in_round

            # ============================== #
            #          Train Step            #
            # ============================== #
            self._update_network(moves_in_round)

            # Advance to next state
            if tot_num_moves >= self._T:
                logging.debug("Maximum number of moves exceeded")
                break
            if t.a.is_game_over() or t.s.is_game_over(allow_move_cycle=True):
                # Print the color of the winning player
                if t.a.is_game_over():
                    msg, winner = "Flag was attacked", t.a.piece.color.name
                elif not t.s.has_next_any_moves():
                    next_color = t.s.next_color.name
                    n = t.s.num_next_moveable_pieces()
                    logging.debug("Losing player %s has %d moveable pieces", next_color, n)
                    msg, winner = "Other player had no moves", t.s.get_winner().name
                else:
                    msg, winner = "Unknown termination condition", "Unknown"
                logging.debug("Episode %d: Winner is %s", self._episode, winner)
                logging.debug("Victory Condition: %s", msg)
                break
        f_out.close()

        logging.debug("Episode %d: Number of Total Moves = %d", self._episode, tot_num_moves)
        logging.debug("Episode %d: Number of Random Moves = %d", self._episode, num_rand_moves)
        f_rand = num_rand_moves / tot_num_moves
        logging.debug("Episode %d: Frac. Moves Random = %.4f", self._episode, f_rand)
        logging.info("COMPLETED episode %d of %d", self._episode, self._M)

    def _fill_initial_move_buffer(self, s_0: State, num_episodes: int = 500, max_moves: int = 700):
        r"""
        Fill move buffer with initially fully random moves.  The function supports serializing the
        move buffer to disk so it does not need to be recreated on each run.
        """
        export_path = SERIALIZE_DIR / ("_initial_move_buffer_%04d.pk" % num_episodes)
        if export_path.exists():
            msg = "Loading initial move buffer: \"%s\"" % export_path
            logging.info("Starting: %s" % msg)
            # Serialize the initial move buffer
            with open(export_path, "rb") as pk_in:
                self._replay = pickle.load(pk_in)
            logging.info("COMPLETED: %s" % msg)
            return

        for i in range(1, num_episodes + 1):
            t = ReplayStateTuple(s=copy.deepcopy(s_0),
                                 base_tensor=DeepQAgent._build_base_tensor(s_0.board, self.d_in))
            tot_num_moves = num_rand_moves = 0
            while True:
                t, moves_in_round, rand_moves_in_round = self._play_moves(t, None)
                tot_num_moves += moves_in_round
                num_rand_moves += rand_moves_in_round

                if tot_num_moves >= max_moves: break
                if t.a.is_game_over() or t.s.is_game_over(allow_move_cycle=True): break

            if i % 20 == 0:
                msg = "Initial population of random replay buffer step %d of %d" % (i, num_episodes)
                logging.info("COMPLETED: %s", msg)

        self._replay.make_all_s_p()
        SERIALIZE_DIR.mkdir(parents=True, exist_ok=True)
        # Serialize the initial move buffer
        with open(export_path, "wb") as pk_out:
            pickle.dump(self._replay, pk_out)

    def _play_moves(self, t: ReplayStateTuple, f_out) -> Tuple[ReplayStateTuple, int, int]:
        r"""
        Advance the environment and add up to /p _MAX_NUM_CONSECUTIVE_MOVES to the replay buffer.

        :param t: Active state tuple
        :param f_out: File output stream where moves are being written for logging purposes
        :return: Tuple of the updated environment state, total number of moves made in this call,
                 and the number of moves made that were random
        """
        moves_in_round = num_rand_moves = 0
        for moves_in_round in range(1, DeepQAgent._MAX_NUM_CONSECUTIVE_MOVES + 1):
            # Prevent effects of cyclic moves and non-Markovian representation causing a false
            # loss condition
            if t.s.is_next_moves_unavailable(): t.s.partial_empty_movestack()
            # If no moves whatsoever, then game over no action defined
            if not t.s.has_next_any_moves(): break
            # With probability < \epsilon, select a random action
            if random.random() < self._eps:
                t.a = t.s.next_player.get_random_move()
                num_rand_moves += 1
            # Select (epsilon-)greedy action
            else:
                _, t.a = self._get_state_move(t.s, t.base_tensor)

            if f_out is not None:
                f_out.write("\n%s,%s,%s" % (t.a.piece.color.name, str(t.a.orig), str(t.a.new)))
                f_out.flush()

            # Player about to win
            if t.a.is_game_over(): t.r = self._WIN_REWARD
            # Game not over yet
            else: t.r = self._NON_TERMINAL_MOVE_REWARD

            t.compress_movestack()
            self._replay.add(t)
            # Advance to next state
            if t.a.is_game_over(): break
            t.s.update(t.a)
        return t, moves_in_round, num_rand_moves

    def _update_network(self, num_batches_to_update: int):
        r"""
        Samples from the replay buffer and updates the network \p num_batches_to_update times

        :param num_batches_to_update: Number of batch updates to perform in this function call
        """
        for _ in range(num_batches_to_update):
            j_arr = self._replay.get_random()
            # Mini-batch support
            reward_arr = [j.r for j in j_arr]
            y_j = torch.cat(reward_arr, dim=0)
            q_j = torch.zeros_like(y_j, dtype=TensorDType)
            for idx, j in enumerate(j_arr):
                q_j[idx] = self._get_action_output_score(j)
                if j.is_terminal(): continue

                # Only time update to save time
                j.create_s_p()
                # If opponent has no moves, overwrite reward as true reward not visible until update
                if not j.s_p.has_next_any_moves():
                    y_j[idx] = j.r = DeepQAgent._WIN_REWARD
                    continue
                # Get next move
                _, a_p = self._get_state_move(j.s_p, j.base_tensor)
                # If other play won, overwrite as true reward not originally visible
                # No need to make move guaranteed to end the game so just skip update/rollback
                if a_p.is_game_over():
                    y_j[idx] = j.r = DeepQAgent._LOSS_REWARD
                    continue

                j.s_p.update(a_p)
                if j.s_p.is_game_over(allow_move_cycle=True):
                    y_j[idx] = j.r = DeepQAgent._LOSS_REWARD
                else:
                    if j.s_p.is_next_moves_unavailable():
                        j.s_p.partial_empty_movestack()
                    with torch.no_grad():
                        y_j_1_val, _ = self._get_state_move(j.s_p, j.base_tensor)
                    y_j[idx] = y_j[idx] + self._gamma * y_j_1_val
                j.s_p.rollback()
            self._optim.zero_grad()
            loss = self._f_loss(y_j, q_j)
            loss.backward()
            self._optim.step()

    def _compare_head_to_head(self, s0: State):
        r""" Test if the current agent is better head to head than previous one """
        temp_back_up = EXPORT_DIR / "_temp_train_backup.pth"
        utils.save_module(self, temp_back_up)

        msg = "Head to head agent competition for episode %d" % self._episode
        logging.debug("Starting: %s", msg)

        backup_epsilon = self._eps
        max_move = num_wins = 0
        cur_flag_att = prev_flag_att = 0
        for _ in range(DeepQAgent._NUM_HEAD_TO_HEAD_GAMES):
            game = Game(self._brd, copy.deepcopy(s0), None)
            if random.random() < 0.5:
                cur_col, prev_col = game.red, game.blue
            else:
                cur_col, prev_col = game.blue, game.red

            cur = DeepQAgent(game.board, cur_col, game.state, disable_import=True)
            utils.load_module(cur, temp_back_up)
            cur._make_rand_move_prob = cur._eps

            prev = DeepQAgent(game.board, prev_col, game.state, disable_import=True)
            utils.load_module(prev, DeepQAgent._TRAIN_BEST_MODEL)
            prev._make_rand_move_prob = cur._eps  # Use same randomness so fair comparison

            winner, flag_attacked = game.two_agent_automated(cur, prev, wait_time=0,
                                                             max_num_moves=4000, display=False)
            if winner == cur.color:
                num_wins += 1
                if flag_attacked: cur_flag_att += 1
            elif flag_attacked:
                prev_flag_att += 1
            elif winner is None:
                max_move += 1

        max_timeouts_to_consider = 5
        denom = DeepQAgent._NUM_HEAD_TO_HEAD_GAMES - min(max_timeouts_to_consider, max_move)
        win_freq = num_wins / denom
        n = DeepQAgent._NUM_HEAD_TO_HEAD_GAMES
        logging.debug("Episode %d: Total Number Games %d", self._episode, n)
        logging.debug("Episode %d: Head to head win frequency %.3f", self._episode, win_freq)
        f_max = max_move / DeepQAgent._NUM_HEAD_TO_HEAD_GAMES
        logging.debug("Episode %d: Halted due to max moves frequency %.3f", self._episode, f_max)
        tot_flag_att = cur_flag_att + prev_flag_att
        logging.debug("Episode %d: Total flag attacks: %d", self._episode, tot_flag_att)
        logging.debug("Episode %d: Current Deep-Q flag attacks: %d", self._episode, cur_flag_att)
        logging.debug("Episode %d: Previous Deep-Q flag attacks: %d", self._episode, prev_flag_att)

        if 0.5 <= win_freq:
            logging.debug("Head to Head: Backing up new best model")
            utils.save_module(self, DeepQAgent._TRAIN_BEST_MODEL)
        else:
            logging.debug("Head to Head: Restore previous best model")
            utils.load_module(self, DeepQAgent._TRAIN_BEST_MODEL)
            self._eps = backup_epsilon
        self.train()
        logging.debug("COMPLETED: %s", msg)

    # def _punish_invalid_move(self, state_tuple: ReplayStateTuple) -> None:
    #     r"""
    #     Updates the network to punish for illegal moves
    #
    #     :param state_tuple: State tuple to be zeroed out
    #     """
    #     _, policy, _, _ = self._get_state_move(state_tuple)
    #     nulled_policy = self._null_blocked_moves(state_tuple, policy, clone=True)
    #     p = self._f_loss(policy, nulled_policy)
    #     self._optim.zero_grad()
    #     p.backward()
    #     self._optim.step()

    # def _num_scout_moves(self) -> int:
    #     r"""
    #     Maximum number of moves a scout can perform. This function deliberately does not consider
    #     the case where some number of moves may be blocked due an obstruction within the board
    #     (i.e, not the board edge).
    #     """
    #     return (self._brd.num_cols - 1) + (self._brd.num_rows - 1)

    def _get_state_move(self, s: State, base_tensor: torch.Tensor) -> Tuple[Tensor, Move]:
        r"""
        Gets the move corresponding to the specified state

        :return: Tuple of the output node, entire policy tensor, maximum valued error, and the
                 selected move respectively
        """
        x = DeepQAgent._build_input_tensor(base_tensor, s.pieces(), s.next_player)
        return self._get_next_move(s, x)

    # def _build_invalid_move_set(self, state_tuple: ReplayStateTuple) -> List[int]:
    #     r"""
    #     Constructs the list of output nodes that do not represent valid moves in the specified
    #     state
    #
    #     :param state_tuple: State tuple to be used
    #     :return: List of invalid move nodes
    #     """
    #     valid = {self._get_output_node_from_move(m) for m in state_tuple.s.next_player.move_set}
    #     cyclic = state_tuple.s.get_cyclic_move()
    #     valid -= {self._get_output_node_from_move(m) for m in cyclic}
    #     return [i for i in range(self._d_out) if i not in valid]

    # def _null_blocked_moves(self, state_tuple: ReplayStateTuple,
    #                         policy: Tensor, clone: bool = False) -> Tensor:
    #     """
    #     Sets all moves that are blocked to the minimum value so they are no selected
    #     :param state_tuple: State tuple to reference
    #     :param policy: Policy tensor from Deep-Q network
    #     :param clone: If True, do not perform null in place
    #     :return: Updated policy tensor with blocked nodes
    #     """
    #     blocked_nodes = self._build_invalid_move_set(state_tuple)
    #     if clone:
    #         policy = policy.clone()
    #     policy[:, blocked_nodes] = DeepQAgent._LOSS_REWARD
    #     return policy

    def _get_move_from_output_node(self, output_node: Union[Tensor, int],
                                   plyr: Player, other: Player) -> Move:
        r"""
        Gets the move for the current player based on the node with the highest value

        :param output_node: Index of the output node
        """
        return self._convert_output_node_to_move(output_node, plyr, other)

    def _get_action_output_score(self, state_tuple: ReplayStateTuple) -> Tensor:
        r"""
        Used in training the agent.  Returns the output score for action :math:`a_j` in :math:`s_j`

        :param state_tuple: State at step :math:`j`
        :return:
        """
        x = self._build_input_tensor(state_tuple.base_tensor, state_tuple.s.pieces(),
                                     state_tuple.s.next_player)
        y = self.forward(x)
        # Extract the scores for specifically the selected piece and move
        piece_loc_idx, move_idx = self._get_loc_and_idx_from_move(state_tuple)
        ret_val = torch.empty((x.shape[0], 2), dtype=TensorDType)
        ret_val[:, 0] = y[:, piece_loc_idx]
        ret_val[:, 1] = y[:, move_idx]
        return ret_val

    @staticmethod
    def _get_loc_and_idx_from_move(state_tuple: ReplayStateTuple) -> Tuple[int, int]:
        r"""
        Given a move, get the corresponding location and index of the move information.

        :param state_tuple: State tuple
        :return: Tuple of the board location index and the move information index
        """
        brd, a = state_tuple.s.board, state_tuple.a
        idx = a.orig.r * brd.num_cols + a.orig.c

        r_diff, c_diff = (a.new.r - a.orig.r), (a.new.c - a.orig.c)
        assert (c_diff == 0 or r_diff == 0) and (c_diff != 0 or r_diff != 0)
        # Any piece other than a scout
        if a.piece.rank != Rank.scout():
            # Get move direction
            assert abs(r_diff) + abs(c_diff) == 1
            for i, move_dir_en in enumerate(Move.Direction.all()):
                if (r_diff, c_diff) == move_dir_en.value:
                    return idx, brd.num_loc + i
            raise ValueError("Should never reach this point")

        # Handle scout moves
        assert abs(r_diff) < brd.num_rows and abs(c_diff) < brd.num_cols
        offset = brd.num_loc + Move.Direction.count()
        # Scout move UP
        if r_diff < 0:
            return idx, offset + (-r_diff - 1)
        offset += brd.num_rows - 1
        # Source move RIGHT
        if c_diff > 0:
            return idx, offset + (c_diff - 1)
        offset += brd.num_cols - 1
        # Scout move DOWN
        if r_diff > 0:
            return idx, offset + (r_diff - 1)
        offset += brd.num_rows - 1
        # Scout move LEFT
        return idx, offset + (-c_diff - 1)

    def _construct_network(self):
        r""" Constructs the neural network portions of the agent. """
        self._q_net = nn.Sequential(_build_conv_block(self.d_in, ResBlock.NUM_PLANES))
        for i in range(self._NUM_RES_BLOCKS):
            self._q_net.add_module("Res_%02d" % i, ResBlock())

        self._head_policy = _build_policy_head(ResBlock.NUM_PLANES, out_planes=2,
                                               board_dim=self._brd.dim(), out_dim=self.d_out)
        # self._head_value = _build_value_head(ResBlock.NUM_PLANES, out_planes=1,
        #                                      board_dim=self._brd.dim())

    def get_next_move(self) -> Move:
        r"""
        Select the next move to be played.
        :return: Move to be performed
        """
        if self._make_rand_move_prob is not None and random.random() < self._make_rand_move_prob:
            return self._plyr.get_random_move()

        x = DeepQAgent._build_input_tensor(self._base_in, self._state.pieces(),
                                           self._state.next_player)
        _, move = self._get_next_move(self._state, x)
        return move

    def _get_next_move(self, state: State, x: Tensor) -> Tuple[Tensor, Move]:
        y_out = self.forward(x)

        # policy = self._null_blocked_moves(state_tuple, , clone=True)
        y = DeepQAgent._null_invalid_locations(state, x, y_out.clone())
        board_loc = int(y[:, :state.board.num_loc].argmax(dim=1))

        ret_tensor = torch.empty((x.shape[0], 2), dtype=TensorDType)
        ret_tensor[:, 0] = y_out[:, board_loc]

        # Location of the piece to move
        orig = Location(board_loc // self._brd.num_cols, board_loc % self._brd.num_cols)
        piece = state.next_player.get_piece_at_loc(orig)
        if piece.is_scout():
            y = self._null_scout_moves(state, piece, y)
            ret_tensor[:, 1], m = self._get_best_scout_move(state, piece, y)
        else:
            # Handle non-scout piece movements
            y = DeepQAgent._null_invalid_directions(state, piece, y)
            ret_tensor[:, 1], m = DeepQAgent._get_best_adjacent_move(state, piece, y)
        return ret_tensor, m

    @staticmethod
    def _null_invalid_locations(state: State, x: Tensor, y: Tensor) -> Tensor:
        r"""
        Null out any locations in \p y where the current player does not have a movable piece.

        :param state: Current state of the game
        :param x: Input vector to the neural network
        :param y: Q vector
        :return: Modified version of \p y such that locations where no move is possible has its
                 valued set to less than the minimum.
        """
        # Find the location of the player's pieces
        piece_locs = torch.sum(x[:, :Rank.moveable_count()], dim=1)
        # Convert piece locations to invalid locations
        # Empty spaces have -1 and empty locations have 1
        # noinspection PyUnresolvedReferences
        null_vec = piece_locs.neg().add(DeepQAgent._PIECE_VAL).mul(DeepQAgent._INVALID_FILL_VAL)
        for p in state.next_player.pieces():
            if not state.piece_has_move(p):  # cyclic check
                null_vec[:, p.loc.r, p.loc.c] = DeepQAgent._INVALID_FILL_VAL
        # Convert from matrix to vector
        null_vec = null_vec.view((null_vec.shape[0], -1))
        # Null out the locations
        y[:, :state.board.num_loc] = y[:, :state.board.num_loc] + null_vec
        return y

    @staticmethod
    def _null_invalid_directions(state: State, p: Piece, y: Tensor) -> Tensor:
        r""" For non-scout pieces, null out the locations that are valid movements """
        plyr = state.next_player
        # Valid move directions
        has_loc = [False] * Move.Direction.count()
        for i, d in enumerate(Move.Direction.all()):
            m = plyr.get_move(p, p.loc.relative(*d.value))
            if m is not None and plyr.is_valid_next(m) and state.is_not_cyclic(m):
                has_loc[i] = True

        # Build null vector
        new_val = DeepQAgent._INVALID_FILL_VAL
        val = [0 if x else new_val for x in has_loc]
        null_vec = torch.tensor(val, dtype=TensorDType)
        # Null out the locations
        start = state.board.num_loc
        y[:, start:start + Move.Direction.count()] += null_vec
        return y

    def _null_scout_moves(self, state: State, p: Piece, y: Tensor) -> Tensor:
        r""" Null out moves that are invalid for the scout """
        assert p.rank == Rank.scout()
        null_vec = torch.full((self._tot_num_scout_moves,), fill_value=DeepQAgent._INVALID_FILL_VAL,
                              dtype=TensorDType)

        offset = 0
        for edge_dir, edge_list in zip(ToEdgeLists.order(), self._brd.to_edge_lists(p.loc)):

            for i, new_loc in enumerate(edge_list):
                m = state.next_player.get_move(p, new_loc)
                if m is None or not state.next_player.is_valid_next(m):
                    break
                # Move needs to be both non-cyclic and valid to not be null-ed out
                if state.is_not_cyclic(m):
                    null_vec[i+offset] = 0
            # Update the offset
            if edge_dir in (ToEdgeLists.UP, ToEdgeLists.DOWN):
                offset += self._brd.num_rows
            else:
                offset += self._brd.num_cols
            offset -= 1  # Adjust offset since one less than number of spaces in each direction
        assert offset == self._tot_num_scout_moves

        # Null out the locations
        offset = state.board.num_loc + Move.Direction.count()
        y[:, offset:] += null_vec
        return y

    @staticmethod
    def _get_best_scout_move(state: State, piece: Piece, y: Tensor) -> Tuple[Tensor, Move]:
        r"""
        Given the selected scout piece that will move, determine the movement.  The movement Q
        vector will be used to select the move itself.  For this portion of \p the vector is
        formatted as: Up in range(1,#Rows-1), Right in range(1,#Cols-1), Down in range(1,#Rows-1),
        and left Right in range(1,#Cols-1).

        :param state: State of the piece
        :param piece: Scout that will move
        :param y: Q vector
        :return: Tuple of the move's Q value and the selected move
        """
        r""" Accessor for the best scout move """
        plyr, loc = state.next_player, piece.loc
        num_row_idx, num_col_idx = state.board.num_rows - 1, state.board.num_cols - 1

        offset = state.board.num_loc + Move.Direction.count()
        y = y[:, offset:]
        idx = int(torch.argmax(y, dim=1))
        ret_val = y[:, idx]
        # Check Up
        if idx < num_row_idx:
            return ret_val, plyr.get_move(piece, loc.relative(-(idx + 1), 0))
        idx -= num_row_idx
        # Check right
        if idx < num_col_idx:
            return ret_val, plyr.get_move(piece, loc.relative(0, idx + 1))
        idx -= num_col_idx
        # Check Down
        if idx < num_row_idx:
            return ret_val, plyr.get_move(piece, loc.relative(idx + 1, 0))
        idx -= num_row_idx
        # Check Left
        assert idx < num_col_idx
        return ret_val, plyr.get_move(piece, loc.relative(0, -(idx + 1)))

    @staticmethod
    def _get_best_adjacent_move(state: State, piece: Piece, y: torch.Tensor) -> Tuple[Tensor, Move]:
        r"""
        Given the selected movable piece that is NOT a scout, determine which move direction
        {Up, Right, Down, Left} (respectively) is the best movement direction.

        :param state: State of the game
        :param piece: Piece that will move
        :param y: Q vector
        :return: Tuple of the value of Q for the selected move and the movement direction
                 respectively.
        """
        start = state.board.num_loc
        # Get best move direction node in vector
        y = y[:, start:start+Move.Direction.count()]
        move_dir = int(torch.argmax(y, dim=1))
        # Convert node location to actual direction
        move_dir_en = Move.Direction.all()[move_dir]
        # noinspection PyArgumentList
        move = state.next_player.get_move(piece, piece.loc.relative(*move_dir_en.value))
        return y[:, move_dir], move

    # def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
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

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Passes the input tensor \p x through the Q network

        :param x: Input tensor
        :return: Tuple of the ten
        """
        # noinspection PyUnresolvedReferences
        assert x.shape[1:] == torch.Size([self.d_in, self._brd.num_rows, self._brd.num_cols])
        y = self._q_net(x)
        return self._head_policy(y)

    def _build_network_input(self, plyr: Player, other: Player) -> Tensor:
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
    def _build_input_tensor(base_in: Tensor, pieces: Iterable[Piece], next_player: Player):
        r"""
        Constructs the tensor to input into the deep Q network.

        :param base_in: Base tensor to which the piece information will be added.  Base tensor
                        has
        :param pieces: Iterable set of all active pieces
        :return: Tensor that can be input into the network
        """
        x = base_in.clone()
        for p in pieces:
            layer_num = DeepQAgent._rank_lookup[p.rank]
            if p.color != next_player.color:
                layer_num += Rank.count()
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


def compare_deep_q_versus_random(brd_file: PathOrStr, state_file: PathOrStr, num_head_to_head: int):
    r"""
    Compares the Deep-Q agent to a random learner

    :param brd_file: Path to the board file
    :param state_file: Path to the state file
    :param num_head_to_head: Number of head to head games
    """
    if not isinstance(brd_file, Path): brd_file = Path(brd_file)
    if not isinstance(state_file, Path): state_file = Path(state_file)

    msg = "Head to head agent competition of Deep-Q versus Random"
    logging.debug("Starting: %s", msg)

    brd = Board.importer(brd_file)
    s0 = State.importer(state_file, brd)

    max_move = num_wins = 0
    deep_q_flag_attack = rand_flag_attack = 0
    for _ in range(num_head_to_head):
        game = Game(brd, copy.deepcopy(s0), None)
        if random.random() < 0.5:
            deep_q_col, rand_col = game.red, game.blue
        else:
            deep_q_col, rand_col = game.blue, game.red

        deep_q_agent = DeepQAgent(game.board, deep_q_col, game.state)
        deep_q_agent._make_rand_move_prob = 0
        rand_agent = RandomAgent(rand_col)

        winner, flag_attacked = game.two_agent_automated(deep_q_agent, rand_agent, wait_time=0,
                                                         max_num_moves=4000, display=False)
        if winner == deep_q_agent.color:
            num_wins += 1
            if flag_attacked: deep_q_flag_attack += 1
        elif flag_attacked:
            rand_flag_attack += 1
        elif winner is None:
            max_move += 1

    win_freq = num_wins / num_head_to_head
    logging.debug("Total Number of Games: %d", num_head_to_head)
    logging.debug("Head to head win frequency %.3f", win_freq)
    logging.debug("Halted due to max moves frequency %.3f", max_move / num_head_to_head)
    logging.debug("Total flag attacks: %d", rand_flag_attack + deep_q_flag_attack)
    logging.debug("Deep-Q flag attacks: %d", deep_q_flag_attack)
    logging.debug("Random flag attacks: %d", rand_flag_attack)
