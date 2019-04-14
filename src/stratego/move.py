import logging
from typing import Optional

from .location import Location
from .piece import Piece


class Move:
    _brd = None

    def __init__(self, p: Piece, orig: Location, new: Location, attacked: Optional[Piece] = None):
        if not Move._is_valid_loc(orig) or not Move._is_valid_loc(new):
            raise ValueError("Location not valid with board")
        if not Move._is_valid_piece(p):
            raise ValueError("Invalid piece to be moved")
        if attacked is not None and p.color == attacked.color:
            raise ValueError("Attacking and attacked pieces cannot be of the same color")

        self._piece, self._attacked = p, attacked
        self._orig, self._new = orig, new

    @staticmethod
    def _is_valid_loc(loc: Location):
        r""" Checks whether a move location is valid """
        if Move._brd is None:
            logging.warning("Trying to verify move location but board information not specified")
            return True
        return Move._brd.is_inside(loc)

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


