import logging
from typing import Optional

from .board import Board
from .location import Location
from .piece import Piece, Rank


class Move:
    _brd = None

    @staticmethod
    def set_board(brd: Board) -> None:
        r""" Specify the \p Board object to be used for verifying moves """
        Move._brd = brd

    def __init__(self, p: Piece, orig: Location, new: Location, attacked: Optional[Piece] = None):
        r"""
        :param p: Piece to move
        :param orig: Piece's original location
        :param new: Piece's new location
        :param attacked: Piece that is being attacked in the move (if any)
        """
        assert orig is not None
        if not Move._is_valid_loc(orig) or not Move._is_valid_loc(new):
            raise ValueError("Location not valid with board")
        # ZSH - This check should be correct but may need to be revisited
        assert p.loc == orig, "Location of piece to moved does not match original location"

        if not Move._is_valid_piece(p):
            raise ValueError("Invalid piece to be moved")

        # ZSH - This check should be correct but may need to be revisited
        assert attacked is None or attacked.loc == new, "Location of attacked mismatch"
        if attacked is not None and p.color == attacked.color:
            raise ValueError("Attacking and attacked pieces cannot be of the same color")

        self._piece, self._attacked = p, attacked
        self._orig, self._new = orig, new

        self.verify()

    @property
    def piece(self) -> Piece:
        r""" Accessor for the piece to be moved """
        return self._piece

    @property
    def attacked(self) -> Piece:
        r""" Returns the piece being attacked (i.e., the one not being moved) """
        assert self._attacked is not None
        return self._attacked

    def is_attack(self) -> bool:
        r""" Returns True if this move corresponds to an attack """
        return self._attacked is not None

    @staticmethod
    def _is_valid_loc(loc: Location):
        r""" Checks whether a move location is valid """
        if Move._brd is None:
            logging.warning("Trying to verify move location but board information not specified")
            return True
        return Move._brd.is_inside(loc)

    def verify(self):
        r""" Simple verifier that the movement is valid """
        assert self._orig != self._new, "Piece cannot move to the same location"

        r_diff, c_diff = self._orig.diff(self._new)
        assert r_diff == 0 or c_diff == 0, "Diagonal moves never allowed"

        man_dist = r_diff + c_diff  # Manhattan distance
        msg = "Only scout can move multiple squares"
        assert man_dist == 1 or self._piece.rank == Rank.scout(), msg

    @staticmethod
    def _is_valid_piece(piece: Piece):
        r""" Checks whether the specified piece is valid for movement """
        if piece is None:
            logging.warning("Piece cannot be None")
            return False
        if piece.is_immobile():
            logging.warning("Trying to move an immobile piece")
            return False
        return True
