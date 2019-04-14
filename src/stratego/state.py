import logging
from enum import Enum
from pathlib import Path
from typing import Union

from stratego import Printer
from .board import Board
from .location import Location
from .move import Move
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

        self._printer = None

    @property
    def next_color(self) -> Color:
        r""" Accessor for the COLOR of the player whose turn is next """
        return self._next

    def toggle_next_color(self) -> None:
        r""" Accessor for the COLOR of the player whose turn is next """
        self._next = Color.RED if self._next == Color.RED else Color.BLUE

    def get_player(self, color: Color) -> Player:
        r""" Get the color associated with the \p Color """
        return self.red if color == Color.RED else self.blue

    @property
    def next_player(self) -> Player:
        r""" Accessor for player whose turn is next """
        return self.red if self._next == Color.RED else self.blue

    @property
    def red(self) -> Player: return self._red

    @property
    def blue(self) -> Player: return self._blue

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
            logging.error("Duplicate piece locations")
            res = False

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

    def update(self, move: Move) -> None:
        if move.piece.color != self.next_color:
            raise ValueError("Move piece color does not match the expected next player")
        # Process the attack
        if move.is_attack():
            self._do_attack(move)
        if not move.is_attack() or move.is_attack_successful():
        # ToDo Ensure update accounts for blocked scout moves
        assert False

        # Switch next player as next
        self.toggle_next_color()

    def get_other_player(self, plyr: Player) -> Player:
        r"""
        Gets other player

        :param plyr: Player whose opposite will be returned
        :return: Red player if \p plyr is blue else the blue player
        """
        return self.red if plyr.color == self.blue else self.blue

    def _do_attack(self, move: Move) -> None:
        r"""
        Process the attack for all data structures in the \p State.  The move is handled separately
        :param move: Attack move
        """
        delete_other = move.is_attack_successful() or move.piece.rank == move.attacked.rank
        # Update the piece information first
        grp = [move.piece]  # Attacker always deleted since move or lose
        # Other piece deleted in tie or successful attack
        if delete_other: grp.append(move.attacked)
        for piece in grp:
            self._printer.delete_piece(piece.loc)
            plyr = self.red if piece.color == Color.RED else self.blue
            plyr.delete_piece_info(piece)

        # Update MoveSet at attacking piece location
        for plyr in [self.red, self.blue]:
            plyr.delete_moveset_info(move.piece.loc, self.get_other_player(plyr))
        # If the attacked piece is removed, update the moveset
        if delete_other:
            other = self.get_player(move.attacked.color)
            other.delete_moveset_info(move.attacked.loc, self.get_other_player(plyr))

    def _do_movement(self, move: Move):
        assert False

    def write_board(self) -> str:
        r""" Return the board contents as a string """
        return self._printer.write()


