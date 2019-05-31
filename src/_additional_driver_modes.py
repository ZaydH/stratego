from stratego import Game, Printer, PathOrStr
from stratego.location import Location
from stratego.move import Move
from stratego.player import Player


# noinspection PyProtectedMember
def _main_hard_coded():
    r""" Player a series of moves for test purposes """
    def _get_move(plyr: Player, l1, l2) -> Move:
        available_moves = plyr.move_set.avail
        values = list(available_moves.values())
        v = [v for v in values if v.orig == Location(*l1) and v.new == Location(*l2)]
        assert v
        return v[0]

    game = Game("boards/small.txt", "states/small_debug.txt", visibility=Printer.Visibility.ALL)
    m = _get_move(game._state.red, (0, 3), (1, 3))
    game.play_move(m, True)
    m = _get_move(game._state.blue, (7, 3), (1, 3))
    game.play_move(m, True)

    game = Game("boards/small.txt", "states/small_debug.txt", visibility=Printer.Visibility.ALL)
    game.display_current()
    m = _get_move(game._state.red, (0, 3), (7, 3))
    game.play_move(m, True)

    game = Game("boards/small.txt", "states/small_debug.txt", visibility=Printer.Visibility.ALL)
    game.display_current()


def _main_make_moves(moves_file: PathOrStr, display_after_move: bool = False):
    game = Game("boards/small.txt", "states/test_debug.txt", visibility=Printer.Visibility.ALL)
    # noinspection PyProtectedMember
    game._execute_move_file(moves_file, display_after_move)
    game.display_current()

