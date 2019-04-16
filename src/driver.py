from stratego import Game
from stratego.printer import Printer
from stratego.random_agent import RandomAgent
from stratego.utils import setup_logger

# ToDo Add argparse


# noinspection PyProtectedMember
def _main_hard_coded():

    from stratego.location import Location
    from stratego.move import Move
    from stratego.player import Player

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
    game.move((0, 1), (1, 1))
    game.display_current()

    game.move((7, 1), (6, 1))
    game.display_current()

    game.move((1, 1), (2, 1))
    game.display_current()

    game.move((6, 1), (5, 1))
    game.display_current()

    game.move((2, 1), (3, 1))
    game.display_current()

    game.move((5, 1), (4, 1))
    game.display_current()

    game.move((3, 1), (4, 1))
    game.display_current()

    game.move((7, 3), (2, 3))
    game.display_current()

    game.move((0, 3), (1, 3))
    game.display_current()

    game.move((2, 3), (2, 5))
    game.display_current()

    game.move((1, 3), (7, 3))
    # game.move((0, 0), (1, 0))
    game.display_current()

    game.move((2, 5), (0, 5))
    game.display_current()

    game.move((7, 3), (7, 2))
    game.display_current()

    game.move((7, 2), (7, 3))
    game.display_current()


def _main_random():
    game = Game("boards/small.txt", "states/small_debug.txt", visibility=Printer.Visibility.ALL)
    # noinspection PyProtectedMember
    a1, a2 = RandomAgent(game._state.red), RandomAgent(game._state.blue)
    game.two_agent_automated(a1, a2, display=True, wait_time=1)


def _main():
    _main_hard_coded()
    _main_random()


if __name__ == "__main__":
    setup_logger()
    _main()
