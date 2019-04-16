from stratego import Game
from stratego.printer import Printer
from stratego.random_agent import RandomAgent
from stratego.utils import setup_logger

# ToDo Add argparse


def _main_hard_coded():
    game = Game("boards/small.txt", "states/small_debug.txt", visibility=Printer.Visibility.ALL)
    game.display_current()

    # from statego.location import Location
    # available_moves = game._state.red.move_set.avail
    # values = list(available_moves.values())
    # v = [v for v in values if v.orig == Location(0, 3) and v.new == Location(7, 3)]
    # assert v
    # v = v[0]
    # game.play_move(v, True)
    # game.display_current()
    # return

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
    a1, a2 = RandomAgent(game._state.red), RandomAgent(game._state.blue)
    game.two_agent_automated(a1, a2, display=True, wait_time=1)


def _main():
    # _main_hard_coded()
    _main_random()


if __name__ == "__main__":
    setup_logger()
    _main()
