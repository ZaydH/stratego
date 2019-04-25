import logging
import random
from typing import Set, Generator, Optional

from .location import Location
from .move import Move
from .piece import Color, Piece, Rank
from .board import Board


class MoveSet:
    _brd = None  # type: Board

    @staticmethod
    def set_board(brd: Board) -> None:
        r""" Sets the board for the entire class """
        MoveSet._brd = brd

    r""" Moves available to a player """
    def __init__(self, color: Color):
        r"""
        :param color: Color that is making the moves
        """
        self._avail = dict()  # Available moves
        self._color = color

    @property
    def avail(self) -> dict:
        r""" Accessor for the available moves """
        return self._avail

    @staticmethod
    def build(pieces: Set[Piece], locs: dict, other_locs: dict) -> 'MoveSet':
        r"""
        Factory method used to construct an initial move set.

        :param pieces: All of the players pieces
        :param locs: Location of the player pieces
        :param other_locs: Location of pieces of other player
        :return: Constructed move set
        """
        assert MoveSet._brd is not None, "Board information be present"
        assert pieces, "Piece set can never be empty"
        color = next(iter(pieces)).color
        ms = MoveSet(color)
        for p in pieces:
            ms.add_piece(p, locs, other_locs)
        return ms

    def add_piece(self, piece: Piece, locs: dict, other_locs: dict):
        r"""
        Add a piece's moves to the MoveSet
        :param piece: Piece whose moves (if any) will be added
        :param locs: Location of the player pieces
        :param other_locs: Location of pieces of other player
        """
        self._process_piece(piece, locs, other_locs, add=True)

    def del_piece(self, piece: Piece, locs: dict, other_locs: dict):
        r"""
        Add a piece's moves to the MoveSet
        :param piece: Piece whose moves (if any) will be added
        :param locs: Location of the player pieces
        :param other_locs: Location of pieces of other player
        """
        self._process_piece(piece, locs, other_locs, add=False)

    def _process_piece(self, piece: Piece, locs: dict, other_locs: dict, add: bool):
        r"""
        Standardizes adding/removing a piece since same algorithm with minor change.

        :param piece: Piece to process
        :param locs: Location for pieces of same color as \p Piece
        :param other_locs: Location for other player's pieces
        :param add: If True, add the piece, otherwise remove the piece
        """
        # Verify color is same for all pieces
        assert piece.color == self._color, "Piece set has pieces of different colors"

        # Standard function for either adding or deleting a move
        def _process_func(_p: Piece, _loc: Location):
            if add:
                try: self._add_move(_p, _loc, other_locs[_loc])
                except KeyError: self._add_move(_p, _loc)
            else: self._del_move(_p, _loc)

        # Bombs and flags can be ignored
        if piece.is_immobile(): return
        # Check ordinary pieces
        if piece.rank != Rank.scout():
            for loc in piece.loc.neighbors():
                # Ignore pieces not allowed by board or where piece of same color
                if not self._brd.is_inside(loc) or loc in locs: continue
                _process_func(piece, loc)
        # Check scout pieces specially
        else:
            for direction_list in self._brd.to_edge_lists(piece.loc):
                for loc in direction_list:
                    # If scout blocked by board location or same color, immediately stop
                    if not self._brd.is_inside(loc) or loc in locs: break
                    _process_func(piece, loc)
                    if loc in other_locs: break

    def _add_move(self, p: Piece, other: Location, attacked: Optional[Piece] = None) -> None:
        r""" Add \p piece's move to \p other to the \p MoveSet """
        assert p.is_scout() or p.loc.is_adjacent(other)

        key = self._make_move_key(p.loc, other)
        # assert key not in self._avail
        self._avail[key] = Move(p, p.loc, other, attacked)

    def _del_move(self, p: Piece, other: Location) -> None:
        r"""
        Delete the corresponding move from the \p MoveSet

        :param p: Piece whose move will be deleted
        :param other: Location where \p p will be moved
        """
        assert p.is_scout() or p.loc.is_adjacent(other)

        key = self._make_move_key(p.loc, other)
        del self._avail[key]

    def has_move(self, p: Piece, new_loc: Location) -> bool:
        r""" Returns True if the \p Piece has an availble move to the specified \p Location """
        key = self._make_move_key(p.loc, new_loc)
        return key in self._avail

    def get_move(self, p: Piece, new_loc: Location) -> Optional[Move]:
        r"""
        Gets the move corresponding to the \p Piece and \p Location.  If the corresponding \p Move
        is not found, \p None is returned.
        """
        key = self._make_move_key(p.loc, new_loc)
        try: return self._avail[key]
        except KeyError: return None

    def __len__(self) -> int:
        r""" Return number of moves in the \p MoveSet """
        return len(self._avail)

    def remove_moves_after_add(self, loc: Location, plyr_locs: dict, other_locs: dict) -> None:
        r"""
        Process the adding of a piece at Location \p loc

        :param loc: Location of added piece
        :param plyr_locs: Location of pieces for same color as \p MoveSet
        :param other_locs: Location of pieces of other \p Player
        """
        self._handle_loc_change(loc, plyr_locs, other_locs, False)

    def add_moves_after_delete(self, loc: Location, plyr_locs: dict, other_locs: dict) -> None:
        r"""
        Process the deletion of a piece that was at Location \p loc

        :param loc: Location of deleted piece
        :param plyr_locs: Location of pieces for same color as \p MoveSet
        :param other_locs: Location of pieces of other \p Player
        """
        self._handle_loc_change(loc, plyr_locs, other_locs, True)

    def _handle_loc_change(self, loc: Location, plyr_locs: dict, other_locs: dict, add: bool):
        r"""
        Process a \p Location's state change by either removing or add moves to the MoveSet.

        :param loc: Location whose state is being changed
        :param plyr_locs: Locations of the implicit player's pieces
        :param other_locs: Location dictionary for the other player
        :param add: If True, add moves to the MoveSet.  Otherwise, remove those locations.
        """
        el = self._brd.to_edge_lists(loc)
        el_groups = [(el.right, el.left), (el.left, el.right), (el.up, el.down), (el.down, el.up)]

        def _add_func(_p: Piece, _loc: Location):
            try: self._add_move(_p, _loc, other_locs[_loc])
            except KeyError: self._add_move(_p, _loc)

        for search, opp in el_groups:
            # Find first piece in search direction (if any)
            p = None
            for srch in search:
                if srch in plyr_locs: p = plyr_locs[srch]
                elif srch in other_locs: p = other_locs[srch]
                if p is not None: break
            # If no piece in search direction
            if p is None or p.is_immobile(): continue
            # Ignore pieces of other color since will be handled in separate function call
            if p.color != self._color: continue
            # If found p is not a scout and not adjacent, move on
            if not p.is_scout() and not p.loc.is_adjacent(loc): continue

            # Delete first since may need to add in next step
            if not add: self._del_move(p, loc)
            # In an add, always add the move.  In a delete, may need to add back if the moved
            # piece is of the other player's color
            if add or loc in other_locs: _add_func(p, loc)

            if p.is_scout():
                for srch in opp:
                    if srch in plyr_locs: break

                    if add: _add_func(p, srch)
                    else: self._del_move(p, srch)

                    # Perform second since could still attack
                    if srch in other_locs: break

    @staticmethod
    def _make_move_key(orig: Location, new: Location):
        return orig, new


class Player:
    r""" Represents one of the two players """
    def __init__(self, color: Color):
        r"""
        :param color: Color of the player
        """
        self._color = color

        # noinspection PyTypeChecker
        self._move_set = None  # type: MoveSet
        self._locs = dict()
        self._pieces = set()

    @property
    def color(self) -> Color:
        r""" Accessor for the \p Player's \p Color. """
        return self._color

    @property
    def num_pieces(self) -> int:
        r""" Accessor for number of pieces the player has """
        return len(self._pieces)

    @property
    def move_set(self) -> MoveSet:
        r""" Accessor for the \p Player's \p MoveSet"""
        return self._move_set

    def add_piece(self, piece: Piece, other: 'Player' = None) -> None:
        r""" Add \p piece to \p Player's set of pieces """
        assert piece not in self._pieces, "Duplicate piece"
        assert piece.loc not in self._locs, "Two pieces in same location"

        self._pieces.add(piece)
        self._locs[piece.loc] = piece

        if other is not None:
            assert self._color != other.color
            self.move_set.add_piece(piece, self._locs, other._locs)

    def delete_piece_info(self, piece: Piece, other: 'Player') -> None:
        r""" Remove \p piece from the \p Player's set of pieces """
        self._pieces.remove(piece)
        del self._locs[piece.loc]
        self.move_set.del_piece(piece, self._locs, other._locs)

    def delete_moveset_info(self, loc: Location, other: 'Player') -> None:
        r""" Update the MoveSet information after deleting a piece at Location \p loc """
        assert self._color != other.color
        self.move_set.add_moves_after_delete(loc, self._locs, other._locs)

    def update_moveset_after_add(self, loc: Location, other: 'Player') -> None:
        r"""
        When adding a piece (i.e., moving it and placing it back down), some previously valid moves
        become blocked.  This method updates \p MoveSet to accomodate that.
        :param loc: \p Location where piece was placed
        :param other: Other player
        """
        assert self._color != other.color
        # pylint: disable=protected-access
        self.move_set.remove_moves_after_add(loc, self._locs, other._locs)

    def has_flag(self) -> bool:
        r""" Returns True if the player has a flag """
        flag = Rank.flag()
        return any(p.rank == flag for p in self._pieces)

    def get_piece_at_loc(self, loc: Location) -> Optional[Piece]:
        r""" Returns the piece at the specified location. If no piece is there, returns None """
        try: return self._locs[loc]
        except KeyError: return None

    def has_move(self, piece: Piece, new_loc: Location) -> bool:
        r""" Returns \p True if the player has a move for the piece ot the specified \p Location """
        assert piece is not None
        return self.move_set.has_move(piece, new_loc)

    def get_move(self, piece: Piece, new_loc: Location) -> Optional[Move]:
        r""" Returns \p True if the player has a move for the piece ot the specified \p Location """
        assert piece is not None
        return self.move_set.get_move(piece, new_loc)

    def piece_locations(self) -> Set[Location]:
        r""" Location of all of the \p Player's pieces """
        set_locs = set(self._locs.keys())
        assert len(set_locs) == len(self._pieces)
        return set_locs

    def pieces(self) -> Generator[Piece, None, None]:
        r""" Generator that yields the Player's pieces """
        for p in self._pieces:
            yield p

    def build_move_set(self, other: 'Player'):
        r""" Construct the move set of the """
        assert self._color != other.color
        self._move_set = MoveSet.build(self._pieces, self._locs, other._locs)

    def verify_piece_set(self, piece_set: Board.PieceSet) -> bool:
        r"""
        Verify that the player piece information is compliance with the \p Board \p PieceSet
        :param piece_set: Piece set maximum counts
        :return: True if the player's piece set information is in compliance
        """
        pieces_by_rank = dict()
        # Count the number of pieces for each rank
        for p in self._pieces:
            try: pieces_by_rank[p.rank] += 1
            except KeyError: pieces_by_rank[p.rank] = 1
        res = True
        for r in Rank.get_all():
            if r in pieces_by_rank and pieces_by_rank[r] > piece_set.get_rank_count(r):
                logging.warning("Color %s has too many pieces of rank: \"%s\"", self._color.name, r)
                res = False
        return res

    def get_random_move(self) -> Move:
        r""" Selects and returns a move uniformly at random from the set of available moves """
        values = list(self.move_set.avail.values())
        return random.choice(values)
