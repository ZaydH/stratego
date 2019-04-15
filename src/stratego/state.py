import logging
from enum import Enum
from pathlib import Path
from typing import Union

from stratego import Printer
from .board import Board
from .location import Location
from .move import Move, MoveStack
from .piece import Color, Piece, Rank
from .player import Player, MoveSet


class State:
    r""" Stores the current state of the game """
    SEP = "|"

    class ImporterKeys(Enum):
        r""" Keys used in the State file """
        next_turn = "NextTurn"
        piece = "Piece"

    def __init__(self, brd: Board):
        # noinspection PyTypeChecker
        self._next = None  # type: Color
        self._brd = brd
        self._red = Player(Color.RED)
        self._blue = Player(Color.BLUE)

        # noinspection PyTypeChecker
        self._printer = None  # type: Printer
        self._stack = MoveStack()

    @property
    def next_color(self) -> Color:
        r""" Accessor for the COLOR of the player whose turn is next """
        return self._next

    def toggle_next_color(self) -> None:
        r""" Accessor for the COLOR of the player whose turn is next """
        self._next = Color.RED if self._next != Color.RED else Color.BLUE

    def get_player(self, color: Color) -> Player:
        r""" Get the color associated with the \p Color """
        return self.red if color == Color.RED else self.blue

    @property
    def next_player(self) -> Player:
        r""" Accessor for player whose turn is next """
        return self.red if self._next == Color.RED else self.blue

    @property
    def red(self) -> Player:
        r""" Accessor for the RED player """
        return self._red

    @property
    def blue(self) -> Player:
        r""" Accessor for the BLUE player """
        return self._blue

    @staticmethod
    # pylint: disable=protected-access
    def importer(file_path: Union[Path, str], brd: Board,
                 vis: Printer.Visibility = Printer.Visibility.NONE) -> 'State':
        r"""
        Factory method to construct \p State objects from a file.

        :param file_path: File defining the current state
        :param brd: Board corresponding to the state to the current state
        :param vis: Specifies whose pieces are visible
        :return: Constructed \p State object
        """
        if isinstance(file_path, str): file_path = Path(file_path)
        try:
            with open(file_path, "r") as f_in:
                lines = f_in.readlines()
        except IOError:
            raise IOError("Unable to read state file \"%s\"" % str(file_path))

        state = State(brd)
        for line in lines:
            line = line.strip()
            if line[0] == "#":
                continue
            spl = line.split(State.SEP)
            # Import next player line
            if spl[0] == State.ImporterKeys.next_turn.value:
                assert len(spl) == 2, "Invalid number entries for next player"
                assert state._next is None, "Duplicate next player line"
                state._next = Color[spl[1].upper()]  # autoconverts string to enum
            elif spl[0] == State.ImporterKeys.piece.value:
                assert len(spl) == 4, "Invalid number of entries for piece line"
                p = Piece(Color[spl[1]], Rank(spl[2]), Location.parse(spl[3]))
                assert brd.is_inside(p.loc)

                plyr = state._red if p.color == Color.RED else state._blue
                plyr.add_piece(p)
            else:
                raise ValueError("Unparseable file file \"%s\"" % line)

        # Define the initial set of moves each player can make
        Move.set_board(brd)
        MoveSet.set_board(brd)
        for plyr, other in [(state._red, state._blue), (state._blue, state._red)]:
            plyr.build_move_set(other)

        # Create the state printer
        state._printer = Printer(state._brd, state.red.pieces(), state.blue.pieces(), vis)

        assert state._is_valid(), "Input state is invalid"
        return state

    def _is_valid(self) -> bool:
        r""" Simple sanity check the state is valid """
        res = True

        # Check no duplicate locations. Intersection of the sets means a duplicate
        if self._red.piece_locations() & self._blue.piece_locations() != set():
            raise ValueError("Duplicate piece locations")

        # Check pieces conform to PieceSet field in Board class
        for plyr in [self._red, self._blue]:
            if not plyr.verify_piece_set(self._brd.piece_set):
                logging.error("Player %s has invalid piece information", plyr.color.name)
                res = False
            if not plyr.has_flag():
                logging.error("Player %s does not have a flag. A flag is required", plyr.color.name)
                res = False

        return res

    @staticmethod
    def _print_template_file(file_path: Union[Path, str]) -> None:
        r"""
        Constructs a template state file for demonstration purposes and to allow the user to fill
        in manually.

        :param file_path: Path where to write the state file
        """
        lines = [["# Specify player whose turn is next"],
                 [State.ImporterKeys.next_turn.value, "RED or BLUE"]]

        for color in [Color.RED.name, Color.BLUE.name]:
            lines.append([f"# Add {color} piece information"])
            for _ in range(3):
                lines.append([State.ImporterKeys.piece.value, color,
                              "RANK", Location.file_example_str()])

        # Write file to disk
        if not isinstance(file_path, Path): file_path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True)
        with open(file_path, "w+") as f_out:
            f_out.write("# State template file generated by \"State._print_template_file\"\n"
                        "# Lines beginning with \"#\" are comments\n")
            f_out.write("\n".join([State.SEP.join(line) for line in lines]))

    def update(self, move: Move) -> bool:
        r"""
        Updates the state of the game by perform the move.

        :param move: Move to be performed
        """
        if move.piece.color != self.next_color:
            raise ValueError("Move piece color does not match the expected next player")

        # ToDo Verify no cyclic moves

        # Check for illegal cycling
        mv_plyr = self.get_player(move.piece.color)
        if not mv_plyr.has_move(move.piece, move.new):
            raise ValueError("Specified move appears does not appear to be known")

        # Delete the piece being moved
        self._printer.delete_piece(move.orig)
        other = self.get_other_player(mv_plyr)
        mv_plyr.delete_piece_info(move.piece, other)

        # Process the attack (if applicable)
        if move.is_attack():
            self._do_attack(move)

        # Regardless of whether move or attack, location information of moving piece needs update
        mv_plyr.delete_moveset_info(move.orig, other)
        other.delete_moveset_info(move.orig, mv_plyr)

        if not move.is_attack() or move.is_attack_successful():
            self._do_piece_movement(move)

        # Switch next player as next
        self.toggle_next_color()
        # Only push onto the move stack once move has definitely been successfully completed
        self._stack.push(move)
        return True

    def get_other_player(self, plyr: Player) -> Player:
        r"""
        Gets other player

        :param plyr: Player whose opposite will be returned
        :return: Red player if \p plyr is blue else the blue player
        """
        return self.red if plyr.color == Color.BLUE else self.blue

    def _do_attack(self, move: Move) -> None:
        r"""
        Process the attack for all data structures in the \p State.  The move is handled separately
        :param move: Attack move
        """
        # Do not use < compare since this will cause runtime error because of how __gt__ checks
        # ranks for stationary pieces
        if move.piece.rank != move.attacked.rank and not move.is_attack_successful():
            return

        other = self.get_player(move.attacked.color)
        # May need to delete the attacked piece's info
        self._printer.delete_piece(move.new)
        other.delete_piece_info(move.attacked, self.next_player)
        other.delete_moveset_info(move.new, self.get_other_player(other))

    def _do_piece_movement(self, move: Move):
        r"""
        Perform the state additions needed just related due the piece in \p move's new location.
        This function is very limited scope and does not handle deleting information from the state.
        """
        # Piece is in the new location
        move.piece.loc = move.new
        plyr = self.get_player(move.piece.color)
        plyr.add_piece(move.piece)

        other = self.get_other_player(plyr)
        plyr.add_moveset_info(move.piece, other)
        plyr.update_moveset_after_add(move.new, other)
        other.update_moveset_after_add(move.new, plyr)

        self._printer.add_piece(move.piece)

    def write_board(self) -> str:
        r""" Return the board contents as a string """
        return self._printer.write()
