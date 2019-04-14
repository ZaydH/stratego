from stratego import Game
from stratego.printer import Printer
from stratego.utils import setup_logger

# ToDo Add argparse


def _main():
    setup_logger()
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


if __name__ == "__main__":
    _main()
