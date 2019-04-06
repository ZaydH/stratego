from stratego import Game, setup_logger


# ToDo Add argparse

def _main():
    setup_logger()
    _ = Game("boards/standard.txt", "states/debug.txt")


if __name__ == "__main__":
    _main()
