from piece import Color, Piece


class Player:
    r""" Represents one of the two players """
    def __init__(self, color: Color):
        self._color = color
        self._pieces = set()

    @property
    def color(self) -> Color:
        r""" Accessor for the \p Player's \p Color. """
        return self._color

    def add_piece(self, piece: Piece) -> None:
        r""" Add \p piece to \p Player's set of pieces """
        assert piece not in self._pieces
        self._pieces.add(piece)

    def remove_piece(self, piece: Piece) -> None:
        r""" Remove \p piece from the \p Player's set of pieces """
        self._pieces.remove(piece)
