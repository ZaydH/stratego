import logging
from typing import Optional

from .board import Board
from .location import Location
from .piece import Piece, Rank


class Move:
    _brd = None
    DISABLE_ASSERT_CHECKS = False

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

        self._piece, self._attacked = p, attacked
        self._orig, self._new = orig, new

        # In some debug or training cases, p may be None, don't crash
        if p is None or Move.DISABLE_ASSERT_CHECKS:
            return

        if not Move._is_valid_loc(orig) or not Move._is_valid_loc(new):
            raise ValueError("Location not valid with board")
        # ZSH - This check should be correct but may need to be revisited
        assert p.loc == orig, "Piece location does not match original location"

        if not Move._is_valid_piece(p):
            raise ValueError("Invalid piece to be moved")

        # ZSH - This check should be correct but may need to be revisited
        assert (not self.is_attack()) or attacked.loc == new, "Location of attacked mismatch"
        if self.is_attack() and p.color == attacked.color:
            raise ValueError("Attacking and attacked pieces cannot be of the same color")

        if not self.verify():
            raise ValueError("Unable to verify move")

    @property
    def piece(self) -> Piece:
        r""" Accessor for the piece to be moved """
        return self._piece

    @property
    def attacked(self) -> Piece:
        r""" Returns the piece being attacked (i.e., the one not being moved) """
        assert self._attacked is not None
        return self._attacked

    @property
    def orig(self) -> Location:
        r""" Accesses the ORIGINAL location of the piece """
        return self._orig

    @property
    def new(self) -> Location:
        r""" Accesses the NEW location of the piece """
        return self._new

    def is_attack(self) -> bool:
        r""" Returns True if this move corresponds to an attack """
        return self._attacked is not None

    def is_move_successful(self) -> bool:
        r"""
        Return True if \p piece successfully moved into location \p new.  This occurs for any move
        that is not an attack or in the case of an attack when \p has a higher rank than
        \p attacked
        """
        return not self.is_attack() or self.is_attack_successful()

    def is_attack_successful(self) -> bool:
        r""" Return True if the moving piece moves into the position of the attacked piece """
        assert self.is_attack()
        return self.piece.rank > self.attacked.rank

    def is_attacked_deleted(self):
        r"""
        Return True if the \p attacked piece is to be deleted.  If it is not an attack, return False
        """
        if not self.is_attack(): return False
        return self.piece.rank >= self.attacked.rank

    def is_game_over(self) -> bool:
        r""" Return True if the move attacked the flag """
        return self.is_attack() and self.attacked.rank == Rank.flag()

    @staticmethod
    def _is_valid_loc(loc: Location):
        r""" Checks whether a move location is valid """
        if Move._brd is None:
            logging.warning("Trying to verify move location but board information not specified")
            return True
        return Move._brd.is_inside(loc)

    def verify(self) -> bool:
        r""" Simple verifier that the movement is valid """
        if self._orig == self._new:
            raise ValueError("Piece cannot move to the same location")

        r_diff, c_diff = self._orig.diff(self._new)
        if r_diff != 0 and c_diff != 0:
            raise ValueError("Diagonal moves never allowed")

        man_dist = r_diff + c_diff  # Manhattan distance
        if man_dist != 1 and self._piece.rank != Rank.scout():
            raise ValueError("Only scout can move multiple squares")
        return True

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

    @staticmethod
    def is_identical(m1: 'Move', m2: 'Move') -> bool:
        r""" Checks whether two moves are identical """
        res = True
        res = res and m1.orig == m2.orig and m1.new == m2.new
        res = res and m1.is_attack() == m2.is_attack()
        if not res:
            return False

        piece_grps = [(m1.piece, m2.piece)]
        if m1.is_attack():
            piece_grps.append((m1.attacked, m2.attacked))
        for p1, p2 in piece_grps:
            res = res and p1.rank == p2.rank and p1.color == p2.color
            res = res and p1.loc == p2.loc
        return res

    # def __eq__(self, other: 'Move') -> bool:
    #     return (isinstance(other, Move) and self.piece == other.piece and self.orig == other.orig
    #             and self.new == other.new and self.attacked == other.attacked)


class MoveStack:
    r""" Checks for cyclic moves """
    def __init__(self):
        self._buf = []

    def top(self) -> Move:
        r""" Returns the element on top of the stack.  Does not affect the stack contents. """
        assert not self.is_empty(), "Move stack empty"
        return self._buf[-1]

    def pop(self) -> Move:
        r""" Remove the \p Move off the top of the stack and return it """
        assert not self.is_empty(), "Move stack empty"
        return self._buf.pop()

    def push(self, move: Move) -> None:
        r""" Place \p move on the top of the stack """
        return self._buf.append(move)

    def is_cyclic(self, move: Move) -> bool:
        # ToDo implement is_cyclic check
        logging.warning("ToDO: is_cyclic check not implemented")
        return False

    def is_empty(self) -> bool:
        r""" Return True if \p MoveStack is empty """
        return not bool(self._buf)

    def __getitem__(self, item: int) -> Move:
        return self._buf[item]

    def __len__(self) -> int:
        r""" Number of elements in the stack """
        return len(self._buf)
