import logging
from typing import Set, Generator

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
    def __init__(self):
        self._avail = dict()  # Available moves

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
        ms = MoveSet()
        for p in pieces:
            # Bombs and flags can be ignored
            if p.is_immobile(): continue
            # Check ordinary pieces
            if p.rank != Rank.scout():
                for loc in p.loc.neighbors():
                    # Ignore pieces not allowed by board or where piece of same color
                    if not ms._brd.is_inside(loc) or loc in locs: continue
                    ms.add(p, loc)
            # Check scout pieces specially
            else:
                for direction_list in ms._brd.to_edge_lists(p.loc):
                    for loc in direction_list:
                        # If scout blocked by board location or same color, immediately stop
                        if not ms._brd.is_inside(loc) or loc in locs: break
                        ms.add(p, loc)
                        if loc in other_locs: break
        return ms

    def add(self, p: Piece, other: Location) -> None:
        r""" Add \p piece's move to \p other to the \p MoveSet """
        key = self._make_move_key(p.loc, other)
        self._avail[key] = Move(p, p.loc, other)

    def __len__(self) -> int:
        r""" Return number of moves in the \p MoveSet """
        return len(self._avail)

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

        self._move_set = None
        self._locs = dict()
        self._pieces = set()

    @property
    def color(self) -> Color:
        r""" Accessor for the \p Player's \p Color. """
        return self._color

    @property
    def move_set(self) -> MoveSet:
        r""" Accessor for the \p Player's \p MoveSet"""
        return self._move_set

    def add_piece(self, piece: Piece) -> None:
        r""" Add \p piece to \p Player's set of pieces """
        assert piece not in self._pieces, "Duplicate piece"
        assert piece.loc not in self._locs, "Two pieces in same location"

        self._pieces.add(piece)
        self._locs[piece.loc] = piece

    def delete_piece(self, piece: Piece) -> None:
        r""" Remove \p piece from the \p Player's set of pieces """
        self._pieces.remove(piece)

    def has_flag(self) -> bool:
        r""" Returns True if the player has a flag """
        flag = Rank.flag()
        return any(p.rank == flag for p in self._pieces)

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
