import sys

from stratego import Game
from stratego.deep_q_agent import DeepQAgent, compare_deep_q_versus_random
# noinspection PyUnresolvedReferences
from stratego.human_agent import HumanAgent
from stratego.printer import Printer
# noinspection PyUnresolvedReferences
from stratego.random_agent import RandomAgent
from stratego.utils import setup_logger, PathOrStr

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


def _main_random():
    game = Game("boards/small.txt", "states/test_debug.txt", visibility=Printer.Visibility.ALL)
    a1, a2 = RandomAgent(game.red), RandomAgent(game.blue)
    # a2 = RandomAgent(game.blue)
    # a1 = HumanAgent(game.red)
    game.two_agent_automated(a1, a2, display=True, wait_time=1)


def _main_deep_q_vs_human():
    game = Game("boards/small.txt", "states/test_debug.txt", visibility=Printer.Visibility.ALL)
    a1 = HumanAgent(game.blue)
    a2 = DeepQAgent(game.board, game.red, game.state)
    game.two_agent_automated(a1, a2, display=True, moves_output_file="moves.txt")


def _main_deep_q_vs_rand():
    compare_deep_q_versus_random("boards/small.txt", "states/test_debug.txt", 101)


def _main_make_moves(moves_file: PathOrStr, display_after_move: bool = False):
    game = Game("boards/small.txt", "states/test_debug.txt", visibility=Printer.Visibility.ALL)
    # noinspection PyProtectedMember
    game._execute_move_file(moves_file, display_after_move)
    game.display_current()


def _main_train():
    game = Game("boards/small.txt", "states/test_debug.txt", visibility=Printer.Visibility.ALL)
    agent = DeepQAgent(game.board, game.red, game.state)
    agent.train_network(game.state)


def _main():
    # _main_hard_coded()
    # _main_random()
    # _main_make_moves("moves.txt")
    # _main_deep_q_vs_rand()
    if len(sys.argv) > 1:
        _main_deep_q_vs_human()
    else:
        _main_train()


if __name__ == "__main__":
    setup_logger()
    _main()
