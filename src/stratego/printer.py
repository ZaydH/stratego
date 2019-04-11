import copy
from enum import Enum

import sty
# from sty import fg, bg, ef, rs, RgbFg

from .location import Location
from .piece import Color, Piece, Rank


class Printer:
    r"""
    Dedicated printer class that caches information about the board so it does not need to be
    regenerated each time.
    """

    SEP = "|"
    EMPTY_LOCATION = " "

    HIDDEN = " "

    IMPASSABLE = chr(9608)  # White box

    class Visibility(Enum):
        NONE = set()
        RED = {Color.RED}
        BLUE = {Color.BLUE}
        ALL = RED | BLUE

    def __init__(self, num_rows: int, num_cols: int, impassable, red_pieces, blue_pieces,
                 visibility: 'Printer.Visibility'):
        r"""
        :param num_rows: Number of rows in the board
        :param num_cols: Number of columns in the board
        :param impassable: Set of board locations not passable by either player
        :param red_pieces: Iterable for the set of red pieces
        :param blue_pieces: Iterable for the set of blue pieces
        :param visibility: Player(s) if any that are visible.
        """
        self._cells = [""]
        base_row = ["\n"] + [Printer.EMPTY_LOCATION for _ in range(num_cols)] + ["\n"]
        for _ in range(num_rows):
            self._cells.append(copy.copy(base_row))
        self._cells.append([""])

        self._visible = visibility.value
        self._row_sep = "-".join(["+"] * (self._n_cols + 1))

        # Fill in the locations that cannot be entered
        self._impassable = {}  # Set dummy value to prevent assertion error
        impass_str = self._impassable_piece()
        for l in impassable:
            self._set_piece(l, impass_str)
        self._impassable = impassable  # Must set after setting cell values to prevent assertion err

        # Add the existing pieces
        for pieces in [red_pieces, blue_pieces]:
            for p in pieces:
                self._set_piece(p.loc, self._format_piece(p), exist_ok=False)

    @property
    def _n_rows(self) -> int:
        r""" Number of rows in the board """
        return len(self._cells) - 2  # Two buffer cells due to printing requirements

    @property
    def _n_cols(self) -> int:
        r""" Number of columns in the board """
        # Two buffer cells due to printing requirements
        # Use second row since first row is filler empty string
        return len(self._cells[1]) - 2

    def _get_piece(self, loc: Location) -> str:
        r""" Get the string for the piece at the specified \p Location """
        self._verify_piece_loc(loc)
        return self._cells[loc.r + 1][loc.c + 1]

    def _set_piece(self, loc: Location, value: str, exist_ok: bool = True):
        r""" Set the string for the piece at the specified \p Location with \p value """
        self._verify_piece_loc(loc)
        if not exist_ok:
            assert self._is_loc_empty(loc), "Setting a location that should be empty"
        self._cells[loc.r + 1][loc.c + 1] = value

    def _verify_piece_loc(self, loc: Location) -> None:
        r""" Verifies whether the piece location is inside the board boundaries"""
        assert loc.is_inside_board(self._n_rows, self._n_cols), "Location outside board dimensions"
        assert loc not in self._impassable, "Trying to get an impassable piece"

    def _is_loc_empty(self, loc: Location) -> bool:
        r""" Returns true if the specified location is empty """
        return self._get_piece(loc) == Printer.EMPTY_LOCATION

    def delete_piece(self, loc: Location) -> None:
        r""" Deletes piece at the specified location """
        assert not self._is_loc_empty(loc), "Tried to delete piece that does not exist"
        self._set_piece(loc, Printer.EMPTY_LOCATION)

    def move_piece(self, orig: Location, new: Location) -> None:
        r"""
        Moves piece from \p orig to \p new.  If \p new has a piece already, that piece is removed
        in the process of the move.
        """
        assert not self._is_loc_empty(orig), "Tried to delete piece that does not exist"
        self._set_piece(new, self._get_piece(orig))
        self.delete_piece(orig)

    def write(self) -> str:
        r""" Prints the board to a large string """
        return self._row_sep.join([Printer.SEP.join(x) for x in self._cells])

    # noinspection PyProtectedMember
    def _is_visible(self, color: Color) -> bool:
        r"""
        Returns True if the piece color is visible
        >>> p = Printer(4,4,{},{},{},Printer.Visibility.NONE)
        >>> print(p._is_visible(Color.RED), p._is_visible(Color.BLUE))
        False False
        >>> p = Printer(4,4,{},{},{},Printer.Visibility.RED)
        >>> print(p._is_visible(Color.RED), p._is_visible(Color.BLUE))
        True False
        >>> p = Printer(4,4,{},{},{},Printer.Visibility.ALL)
        >>> print(p._is_visible(Color.RED), p._is_visible(Color.BLUE))
        True True
        """
        return color in self._visible

    def _format_piece(self, piece: Piece) -> str:
        r""" Generates the string for a piece to appear in the output """
        # white_fg = fg(255, 255, 255)
        white_fg = sty.fg.li_white
        return "".join([sty.rs.all,
                        sty.bg.da_red if piece.color == Color.RED else sty.bg.blue,
                        sty.ef.bold, white_fg,  # White writing over the background
                        str(piece.rank) if self._is_visible(piece.color) else Printer.HIDDEN,
                        sty.rs.all  # Go back to normal printing
                        ])

    def _impassable_piece(self) -> str:
        r""" Generates string for a square that is impassable """
        white_bg = sty.bg.li_white
        white_fg = sty.fg.li_white
        return "".join([sty.rs.all, white_bg,
                        sty.ef.bold, white_fg,  # White writing over the background
                        Printer.IMPASSABLE,
                        sty.rs.all  # Go back to normal printing
                        ])
