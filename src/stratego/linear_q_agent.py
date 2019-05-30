import sys
from enum import Enum, auto

import torch
import torch.nn as nn
from torch import Tensor
from typing import Iterable

from . import State, Player, Board
from .deep_q_agent import DeepQAgent
from .piece import Rank, Piece
from .utils import EXPORT_DIR


class LinearVectorFeature(Enum):
    ORIG_ROW = auto()
    ORIG_COL = auto()
    NEW_ROW = auto()
    NEW_COL = auto()
    RANK_START = auto()
    DIST_TO_FLAG = auto() + 2 * Rank.count()


class LinearQAgent(DeepQAgent):

    _TRAIN_BEST_MODEL = EXPORT_DIR / "_linear_q_checkpoint_best_model.pth"
    _EXPORTED_MODEL = EXPORT_DIR / "final_linear_q.pth"

    def __init__(self, brd: Board, plyr: Player, state: State, eps_end: float = 1e-4,
                 disable_import: bool = False):
        r"""
        :param brd: Game board
        :param plyr: Player who will be controlled by the agent.
        :param disable_import: Disable importing of an agent from disk
        """
        super().__init__(brd, plyr, state, eps_end, disable_import)

    @staticmethod
    def _get_initial_in_dim() -> int:
        r""" Set value of field \p _d_in """
        return 2 * Rank.count() + 2

    def _construct_network(self):
        r""" Construct the input network """
        self._seq = nn.Sequential(nn.Linear(self._d_in, self._d_out),
                                  nn.Tanh())

    def forward(self, x):
        r""" Simple linear model """
        return self._seq(x)

    @staticmethod
    def _find_min_flag_distance(flag: Piece, pieces: Iterable[Piece]):
        min_dist = sys.maxsize
        for p in pieces:
            if p.is_immobile() or p.color != flag.color: continue
            dist = p.loc - flag.loc
            min_dist = min(min_dist, dist)
        return min_dist

    @staticmethod
    def _build_input_tensor(base_in: Tensor, pieces: Iterable[Piece], next_player: Player):
        d_in = LinearQAgent._get_initial_in_dim()
        x = torch.zeros((next_player.move_set, d_in))

        v_in_base = torch.zeros()
        # Find the flags
        next_flag = other_flag = None
        for p in pieces:
            if p.rank == Rank.flag():
                if p.color == next_player.color: next_flag = p
                else: other_flag = p
            else:
                idx = LinearQAgent._rank_lookup[p.rank]
                if p.rank != next_player.color: idx += Rank.count()
                v_in_base[idx] += 1

        # Distance to flag features
        idx = LinearVectorFeature.DIST_TO_FLAG.value
        v_in_base[idx] = LinearQAgent._find_min_flag_distance(next_flag, pieces)
        v_in_base[idx+1] = LinearQAgent._find_min_flag_distance(other_flag, pieces)

        moves_vec = []
        for m in next_player.move_set.avail:






