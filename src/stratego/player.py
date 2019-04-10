import logging
from typing import Set, Generator

from .location import Location
from .piece import Color, Piece, Rank
from .board import Board


class Player:
    r""" Represents one of the two players """
    def __init__(self, color: Color):
        self._color = color
        self._locs = dict()
        self._pieces = set()

    @property
    def color(self) -> Color:
        r""" Accessor for the \p Player's \p Color. """
        return self._color

    def add_piece(self, piece: Piece) -> None:
        r""" Add \p piece to \p Player's set of pieces """
        assert piece not in self._pieces, "Duplicate piece"
        assert piece.loc not in self._locs, "Two pieces in same location"

        self._pieces.add(piece)
        self._locs[piece.loc] = piece

    def delete_piece(self, piece: Piece) -> None:
        r""" Remove \p piece from the \p Player's set of pieces """
        self._pieces.remove(piece)

    def piece_locations(self) -> Set[Location]:
        r""" Location of all of the \p Player's pieces """
        set_locs = set(self._locs.keys())
        assert len(set_locs) == len(self._pieces)
        return set_locs

    def pieces(self) -> Generator[Piece, None, None]:
        r""" Generator that yields the Player's pieces """
        for p in self._pieces:
            yield p

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
