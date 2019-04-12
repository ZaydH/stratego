from stratego import Game, setup_logger
from stratego.printer import Printer

# ToDo Add argparse


def _main():
    setup_logger()
    game = Game("boards/small.txt", "states/small_debug.txt", visibility=Printer.Visibility.ALL)
    game.display_current()


if __name__ == "__main__":
    _main()
