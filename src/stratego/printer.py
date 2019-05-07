import copy
from enum import Enum

import sty

from .board import Board
from .location import Location
from .piece import Color, Piece


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

    def __init__(self, brd: Board, red_pieces, blue_pieces,
                 visibility: 'Printer.Visibility'):
        r"""
        :param brd: Board information.
        :param red_pieces: Iterable for the set of red pieces
        :param blue_pieces: Iterable for the set of blue pieces
        :param visibility: Player(s) if any that are visible.
        """
        self._brd = brd

        # Needs to be a list of list since later join in write function
        self._cells = [[" ".join(["  "] + [str(i) for i in range(self._brd.num_cols)] + ["\n"])]]
        # Construct list elements that will hold piece info
        base_row = [""] + [Printer.EMPTY_LOCATION for _ in range(self._brd.num_cols)] + ["\n"]
        for i in range(self._brd.num_rows):
            self._cells.append(copy.copy(base_row))
            self._cells[-1][0] = '\n{:2d}'.format(i)
        self._cells.append([""])

        self._visible = visibility.value
        self._row_sep = "".join(["  ", "-".join(["+"] * (self._n_cols + 1))])

        # Fill in the locations that cannot be entered
        impass_str = self._impassable_piece()
        for l in self._brd.blocked:
            self._set_piece(l, impass_str, ignore_impassable=True)

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

    def _set_piece(self, loc: Location, value: str, exist_ok: bool = True,
                   ignore_impassable: bool = False) -> None:
        r""" Set the string for the piece at the specified \p Location with \p value """
        if not ignore_impassable:
            self._verify_piece_loc(loc)
        else:
            assert loc.is_inside_board(self._brd.num_rows, self._brd.num_cols)
        if not exist_ok:
            assert self._is_loc_empty(loc), "Setting a location that should be empty"
        self._cells[loc.r + 1][loc.c + 1] = value

    def _verify_piece_loc(self, loc: Location) -> None:
        r""" Verifies whether the piece location is inside the board boundaries"""
        assert self._brd.is_inside(loc), "Invalid location in board"

    def _is_loc_empty(self, loc: Location) -> bool:
        r""" Returns true if the specified location is empty """
        return self._get_piece(loc) == Printer.EMPTY_LOCATION

    def delete_piece(self, loc: Location) -> None:
        r""" Deletes piece at the specified location """
        assert not self._is_loc_empty(loc), "Tried to delete piece that does not exist"
        self._set_piece(loc, Printer.EMPTY_LOCATION)

    # def move_piece(self, orig: Location, new: Location) -> None:
    #     r""" Move piece from \p orig to \p new """
    #     assert not self._is_loc_empty(orig), "Trying to move an empty location"
    #     piece_str = self._get_piece(orig)
    #     self._set_piece(new, piece_str)
    #     self.delete_piece(orig)

    def add_piece(self, piece: Piece) -> None:
        r""" Add the specified piece to the printer """
        assert self._is_loc_empty(piece.loc), "Trying to add piece to non-empty location"
        self._set_piece(piece.loc, self._format_piece(piece))

    def write(self) -> str:
        r""" Prints the board to a large string """
        return self._row_sep.join([Printer.SEP.join(x) for x in self._cells])

    # noinspection PyProtectedMember
    def _is_visible(self, color: Color) -> bool:
        r""" Returns True if the piece color is visible """
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

    @staticmethod
    def _impassable_piece() -> str:
        r""" Generates string for a square that is impassable """
        white_bg = sty.bg.li_white
        white_fg = sty.fg.li_white
        return "".join([sty.rs.all, white_bg,
                        sty.ef.bold, white_fg,  # White writing over the background
                        Printer.IMPASSABLE,
                        sty.rs.all  # Go back to normal printing
                        ])
