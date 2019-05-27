import itertools
from enum import Enum
from pathlib import Path
from typing import Union, Iterable, Set, Optional

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

        self._is_training = False

    @property
    def is_training(self) -> bool:
        r""" Return \p True if in training mode """
        return self._is_training

    @is_training.setter
    def is_training(self, val: bool) -> None:
        r""" Update the training value """
        self._is_training = val

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
    def other_player(self) -> Player:
        r""" Accessor for the player whose turn is NOT next """
        return self.blue if self._next == Color.RED else self.red

    @property
    def red(self) -> Player:
        r""" Accessor for the RED player """
        return self._red

    @property
    def blue(self) -> Player:
        r""" Accessor for the BLUE player """
        return self._blue

    @property
    def board(self) -> Board:
        r""" Accessor for the board associated with the \p State """
        return self._brd

    def pieces(self) -> Iterable[Piece]:
        r""" Constructs an iterable for the set of all (remaining) pieces in the current \p State"""
        return itertools.chain(self.red.pieces(), self.blue.pieces())

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

        state.red.build_move_set(state.blue)
        state.blue.build_move_set(state.red)

        # Create the state printer
        state._printer = Printer(state._brd, state.red.pieces(), state.blue.pieces(), vis)

        assert state._is_valid(), "Input state is invalid"
        return state

    def _is_valid(self) -> bool:
        r""" Simple sanity check the \p State object is valid """
        # Check no duplicate locations. Intersection of the sets means a duplicate
        if self._red.piece_locations() & self._blue.piece_locations() != set():
            raise ValueError("Duplicate piece locations")

        # Check pieces conform to PieceSet field in Board class
        for plyr in [self._red, self._blue]:
            if not plyr.verify_piece_set(self._brd.piece_set):
                raise ValueError(("Player %s has invalid piece information" % plyr.color.name))
            if not plyr.has_flag():
                raise ValueError("Player %s does not have a flag" % plyr.color.name)
        return True

    def get_cyclic_move(self) -> Set[Move]:
        r""" Gets the move (if any) blocked because it would be cyclic """
        # if self._is_training:
        #     end = -min(13, len(self._stack))
        #     # noinspection PyTypeChecker
        #     invalid_moves = set(self._stack[-2:end:-2])
        #     return invalid_moves
        cyc_min_len = 6
        if len(self._stack) >= cyc_min_len and Move.is_identical(self._stack[-2], self._stack[-6]):
            return {self._stack[-4]}
        return set()

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
            raise ValueError("Specified move does not appear to be known")

        # Delete the piece being moved regardless of whether or not move is an attack
        self._printer.delete_piece(move.orig)
        other = self.get_other_player(mv_plyr)
        mv_plyr.delete_piece_info(move.piece, other)
        self._delete_moveset_info(move.orig)

        # Process the attack (if applicable)
        if move.is_attack():
            self._do_attack(move)

        move.piece.loc = move.new  # Always move the piece even in a loss for consistency
        if move.is_move_successful():
            self._do_piece_movement(move)

        # Switch next player as next
        self.toggle_next_color()
        # Only push onto the move stack once move has definitely been successfully completed
        self._stack.push(move)
        return True

    def _delete_moveset_info(self, loc: Location) -> None:
        r"""
        Regardless of whether move or attack, the set of valid movements MAY change for BOTH
        players.  Therefore, delete moveset information clearing out \p loc

        :param loc: Location of the \p MoveSet information to be deleted
        """
        self.red.delete_moveset_info(loc, self.blue)
        self.blue.delete_moveset_info(loc, self.red)

    def _update_moveset_after_piece_add(self, loc: Location) -> None:
        r"""
        Update the \p MoveSet objects of BOTH players after a piece was placed at location \p loc.

        :param loc: Location of the piece that was added
        """
        self.red.update_moveset_after_add(loc, self.blue)
        self.blue.update_moveset_after_add(loc, self.red)

    def undo(self, num_moves: int = 1):
        r"""
        Undo the last \p num_moves on top of the \p MoveStack

        :param num_moves: Number of moves to rollback.
        """
        if num_moves <= 0:
            raise ValueError("Number of moves to roll back must be a positive integer")
        if num_moves > len(self._stack):
            m = "Cannot undo more moves than on stack: %d > %d" % (num_moves, len(self._stack))
            raise ValueError(m)

        for _ in range(num_moves):
            self.rollback()

    def rollback(self):
        r""" Rollback move on the top of the \p MoveStack """
        m = self._stack.top()
        mv_plyr = self.get_player(m.piece.color)
        other = self.get_other_player(mv_plyr)
        assert self.next_player == other, "Move piece's color should match other player color"

        if m.is_move_successful():
            mv_plyr.delete_piece_info(m.piece, other)
            self._delete_moveset_info(m.new)

        # In case of an attack, need to add back the piece removed
        if m.is_attacked_deleted():
            other.add_piece(m.attacked, mv_plyr)
            self._update_moveset_after_piece_add(m.new)

        m.piece.loc = m.orig
        mv_plyr.add_piece(m.piece, other)
        self._update_moveset_after_piece_add(m.orig)

        # Update the printer
        if m.is_move_successful():
            self._printer.delete_piece(m.new)
        self._printer.add_piece(m.piece)
        if m.is_attacked_deleted():
            self._printer.add_piece(m.attacked)

        # Switch player and update stack only after undo successful
        self.toggle_next_color()
        self._stack.pop()

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
        if not move.is_attacked_deleted():
            return

        other = self.get_player(move.attacked.color)
        # May need to delete the attacked piece's info
        self._printer.delete_piece(move.new)
        other.delete_piece_info(move.attacked, self.next_player)
        # other.delete_moveset_info(move.new, self.next_player)
        self._delete_moveset_info(move.new)

    def _do_piece_movement(self, move: Move):
        r"""
        Perform the state additions needed just related due the piece in \p move's new location.
        This function is very limited scope and does not handle deleting information from the state.
        """
        # Piece is in the new location
        plyr = self.get_player(move.piece.color)

        other = self.get_other_player(plyr)
        plyr.add_piece(move.piece, other)
        self._update_moveset_after_piece_add(move.new)

        self._printer.add_piece(move.piece)

    def is_not_cyclic(self, m: Move) -> bool:
        r""" Check whether the specified move is in the set of cyclic moves"""
        for c_m in self.get_cyclic_move():
            if Move.is_identical(m, c_m):
                return False
        return True

    def piece_has_move(self, p: Piece) -> bool:
        r""" Returns \p True if the piece has any move at all"""
        # Since pieces cant jump only need to check adjacent locs.  No adjacent moves, then
        # definitely has no move at all
        for m in self.get_player(p.color).move_set.avail.values():
            if m.piece.color == p.color and m.orig == p.loc and m.piece.rank == p.rank:
                if self.is_not_cyclic(m):
                    return True
        return False

    def is_game_over(self) -> bool:
        r""" Returns True if the game has ended """
        # Last move attacked flag
        if not self._stack.is_empty() and self._stack.top().is_game_over(): return True
        # Current player has no moves
        return self.next_player.move_set.is_empty(self.get_cyclic_move())

    def get_winner(self) -> Optional[Color]:
        r""" Gets the color of the winning player """
        if not self.is_game_over():
            return None
        if not self._stack.is_empty() and self._stack.top().is_game_over():
            return self._stack.top().piece.color
        if self.next_player.move_set.is_empty(self.get_cyclic_move()):
            return self.other_player.color
        raise RuntimeError("Not able to determine winner")

    def write_board(self) -> str:
        r""" Return the board contents as a string """
        return self._printer.write()
